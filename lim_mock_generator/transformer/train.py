import os
import sys
import argparse
import json

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split
from torch.distributions import Beta
import torch.nn.functional as F

from lim_mock_generator.utils.training_utils import MyDataset
from model import my_model

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--use_dist", action="store_true")
    parser.add_argument("--use_vel", action="store_true")

    # training parameters
    parser.add_argument("--data_path", type=str, default="data.h5")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--save_freq", type=int, default=100)

    # model parameters
    parser.add_argument("--model_name", type=str, default="transformer1")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_features_out", type=int, default=200)

    return parser.parse_args()

def train_model(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ### Load data
    norm_params = np.loadtxt("./norm_params.txt")
    dataset = MyDataset(args.data_path, max_length=args.max_length, norm_params=norm_params, use_dist=args.use_dist, use_vel=args.use_vel)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if args.use_sampler:
        def get_sampler(x, nbins=20, xmin=0, xmax=1):
            bins = torch.linspace(xmin, xmax, steps=nbins+1)
            bin_indices = torch.bucketize(x, bins)
            counts = torch.bincount(bin_indices)
            weights = 1. / counts[bin_indices]
            weights = torch.clamp(weights, min=0.05, max=1) # avoid too small weights
            return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)

        sampler = get_sampler(train_dataset.dataset.x[train_dataset.indices][:,0])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler) 
        sampler = get_sampler(val_dataset.dataset.x[val_dataset.indices][:,0])
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    _, args.num_features_in = train_dataset[0][1].shape
    
    print(f"# Training data: {len(train_dataset)}")
    print(f"# Validation data: {len(val_dataset)}")

    ### Define model
    model = my_model(args)
    model.to(device)
    print(model)

    ### Save arguments
    args.norm_params = norm_params.tolist()
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")

    ### Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)

    def loss_func(output, target, mask):
        # output: (batch, seq_length, num_features_in, num_features_out)
        # target: (batch, seq_length, num_features_in)

        batch_size, seq_length, num_features_in, num_features_out = output.shape
            
        if num_features_out == 1:
            loss = F.mse_loss(output, target, reduction='none')
            #loss = loss.mul(mask).sum() / (mask.sum() + 1e-8)
            loss = loss.mean()
        elif num_features_out == 2:
            alpha, beta = output[:, :, 0], output[:, :, 1]
            beta_dist = Beta(alpha, beta)
            log_prob = beta_dist.log_prob(target)
            loss = - (log_prob * mask).sum() / mask.sum() # ignore padding except for the first one.
        else:
            target_bins = (target * num_features_out).long() # (batch, seq_length, num_features_in) [0, 1] -> [0, num_features_out-1]
            target_bins = torch.clamp(target_bins, min=0, max=num_features_out - 1)

            output = output.view(-1, num_features_out) # (batch * seq_length * num_features_in, num_features_out)
            target_bins = target_bins.view(-1) # (batch * seq_length * num_features_in, )
            mask = mask.view(-1) # (batch * seq_length * num_features_in, )

            log_prob = torch.log(output + 1e-8)
            loss_nll = F.nll_loss(log_prob, target_bins, reduction='none') 
            loss = (loss_nll * mask).sum() / (mask.sum() + 1e-8)
        
        return loss 

    fname_log = f"{args.output_dir}/log.txt"

    with open(fname_log, "w") as f:
        f.write(f"# loss loss_val\n")

    num_batches = len(train_dataloader)
    for epoch in tqdm(range(args.num_epochs)):
        model.train()

        teacher_forcing_ratio = 1.
        #teacher_forcing_ratio = 1. if epoch < 10 else 0.
        #teacher_forcing_ratio = 1. - epoch / args.num_epochs

        for count, (context, seq, mask) in enumerate(train_dataloader):
            context = context.to(device)  # (batch, num_condition)   
            seq = seq.to(device)     # (batch, max_length, num_features_in)
            mask = mask.to(device)   # (batch, max_length)
            #weight = (torch.tensor(10).pow(context[:,0])).unsqueeze(1).unsqueeze(2).expand_as(seq).contiguous() # (batch, max_length, num_features_in)
            
            optimizer.zero_grad()

            input_seq = seq[:, :-1]
            output = model(context, input_seq) # (batch, max_length, num_features_in, num_features_out)
            #_, output = model.generate(context, seq=seq, teacher_forcing_ratio=teacher_forcing_ratio) 
            # output: (batch, max_length, num_features_in, num_features_out)
            
            loss = loss_func(output, seq, mask)

            loss.backward()
            optimizer.step()

            model.eval()
            for context_val, seq_val, mask_val in val_dataloader:
                with torch.no_grad():
                    context_val = context_val.to(device)
                    seq_val = seq_val.to(device)
                    mask_val = mask_val.to(device)
                    #weight = (context_val[:,0].pow(10)).unsqueeze(1).unsqueeze(2).expand_as(seq_val).contiguous() # (batch, max_length, num_features_in)
                    
                    input_seq_val = seq_val[:, :-1]
                    output_val = model(context_val, input_seq_val)
                    loss_val = loss_func(output_val, seq_val, mask_val)
                    
                    break # show one batch result only
            model.train()

            epoch_now = epoch + count / num_batches
            with open(fname_log, "a") as f:
                f.write(f"{epoch_now:.8f} {loss.item():.4f} {loss_val.item():.4f}\n")

        scheduler.step()
        
        # save model
        
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs: 
            fname = f"{args.output_dir}/model_ep{epoch+1}.pth"
            torch.save(model.state_dict(), fname)
            print(f"# Model saved to {fname}")

            fname = f"{args.output_dir}/model.pth"
            torch.save(model.state_dict(), fname)
            print(f"# Model saved to {fname}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
