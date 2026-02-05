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

from cosmoglint.utils import MyDataset, load_global_params
from cosmoglint.model.transformer import transformer_model

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--show_pbar", action=argparse.BooleanOptionalAction, default=True)

    # dataset parameters
    parser.add_argument("--data_path", type=str, nargs='+', default=["data.h5"], help="Path to the data file(s). If the first file contains '*', indices will be used to specify the files, and other files will be ignored.")
    parser.add_argument("--indices", type=str, default=None, help="e.g., 0-999. Only used when data_path contains *.")
    parser.add_argument("--norm_param_file", type=str, default="./norm_params.json")
    parser.add_argument("--global_param_file", type=str, nargs='+', default=None, help="Path to the global parameters file(s). If not None, the number of files must match that of data_path.")

    parser.add_argument("--input_features", type=str, nargs='+', default=["GroupMass"])
    parser.add_argument("--output_features", type=str, nargs='+', default=["SubhaloSFR", "SubhaloDist", "SubhaloVrad", "SubhaloVtan"])
    parser.add_argument("--global_features", type=str, nargs='+', default=None)
    parser.add_argument("--max_length", type=int, default=30)

    # training parameters
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sampler_weight_min", type=float, default=1, help="Minimum weight for the sampler, set to 1 to disable sampling")
    parser.add_argument("--lambda_penalty_loss", type=float, default=0, help="Coefficient for the penalty loss")
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--exclude_ratio", type=float, default=0.0, help="Exclude halos in the corner of a size (exclude_ratio * BoxSize)^3")

    # model parameters
    parser.add_argument("--model_name", type=str, default="transformer1")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_features_out", type=int, default=200)
    parser.add_argument("--use_flat_representation", action=argparse.BooleanOptionalAction, default=True, help="If true, use flattened point features (B, N*M). If false, keep (B, N, M).")

    return parser.parse_args()

def train_model(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print("# Using device: {}".format(device))

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    ### Define model   
    args.num_features_cond = len(args.input_features)
    args.num_features_in = len(args.output_features)
    args.num_features_global = 0 if args.global_features is None else len(args.global_features)
    
    model = transformer_model(args)
    model.to(device)
    print(model)

    ### Load data
    with open(args.norm_param_file) as f:
        norm_param_dict = json.load(f)
    
    global_params = load_global_params(args.global_param_file, args.global_features, norm_param_dict=norm_param_dict)
    
    data_path = args.data_path.copy()
    if "*" in data_path[0] and args.indices is not None:
        # Currently only support one data path with *
        if len(data_path) > 1:
            raise ValueError("When data_path contains *, only one data path is allowed.")
        
        indices = args.indices.split("-")
        istart = int(indices[0])
        iend = int(indices[1])
        print(f"# Using data files from {istart} to {iend}")
        data_path = [ data_path[0].replace("*", str(i)) for i in range(istart, iend+1) ]
        
        if global_params is not None:
            global_params = global_params[istart:iend+1, :]

    dataset = MyDataset(data_path, args.input_features, args.output_features, global_params=global_params, norm_param_dict=norm_param_dict, max_length=args.max_length, exclude_ratio=args.exclude_ratio, use_flat_representation=args.use_flat_representation, show_pbar=args.show_pbar)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def get_sampler(x, nbins=20, xmin=0, xmax=1, temperature=1, weight_min=1e-8):
        bins = torch.linspace(xmin, xmax, steps=nbins+1)
        bin_indices = torch.bucketize(x, bins)
        counts = torch.bincount(bin_indices)
        weights = 1. / counts[bin_indices] 
        weights = weights.pow(temperature) # Apply temperature scaling
        weights = weights.clamp(min=weight_min) # Avoid zero weights
        # When setting replacement to True and num_samples to the original number of samples, the sampler can select the same sample multiple times even within a single epoch.
        # The minimum weight is set to balance the sampling (few samples appear less frequently than when minimum is not set) 
        # Large minimum weight (larger than ~1e-5: the maximum number of halo mass function at z = 2) means the rare samples will be sampled more frequently (could suffer from overfitting, but might be faster to converge)
        return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    
    if args.sampler_weight_min < 1:
        sampler = get_sampler(train_dataset.dataset.x[train_dataset.indices][:,0], weight_min=args.sampler_weight_min)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler) 
        sampler = get_sampler(val_dataset.dataset.x[val_dataset.indices][:,0], weight_min=args.sampler_weight_min)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("# Training data: {:d}".format(len(train_dataset)))
    print("# Validation data: {:d}".format(len(val_dataset)))

    ### Save arguments
    args.norm_param_dict = norm_param_dict
    fname = "{}/args.json".format(args.output_dir)
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print("# Arguments saved to {}".format(fname))

    ### Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    def loss_func(batch, weight=None):

        seq = batch["target"].to(device)     # (batch, max_length, num_features_in)
        mask = batch["mask"].to(device)   # (batch, max_length)
        context = batch["context"].to(device)  # (batch, num_condition)   
        global_cond = batch["global_context"].to(device) # (batch, num_features_global)
        
        input_seq = seq[:, :-1]
        target = seq

        output = model(context, input_seq, global_cond=global_cond) # (batch, max_length, num_features_in, num_features_out)
        #_, output = model.generate(context, seq=seq, teacher_forcing_ratio=teacher_forcing_ratio) 
        # output: (batch, max_length, num_features_in, num_features_out)

        batch_size, seq_length, num_features_in, num_features_out = output.shape
        if weight is None:
            weight = torch.ones_like(target, dtype=torch.float32, device=target.device) # (batch, seq_length)

        weight = mask * weight

        log_prob = torch.log( output + 1e-8 )
        target_bins = (target * num_features_out).long() # (batch, seq_length, num_features_in) [0, 1] -> [0, num_features_out-1]
        target_bins = torch.clamp(target_bins, min=0, max=num_features_out - 1)

        log_prob_flatten = log_prob.contiguous().view(-1, num_features_out) # (batch * seq_length * num_features_in, num_features_out)
        target_bins_flatten = target_bins.contiguous().view(-1) # (batch * seq_length * num_features_in, )
        weight_flatten = weight.contiguous().view(-1) # (batch * seq_length * num_features_in, )

        loss_nll = F.nll_loss(log_prob_flatten, target_bins_flatten, reduction='none') 
        loss = (loss_nll * weight_flatten).sum() / ( (weight_flatten).sum() + 1e-8 )

        if args.lambda_penalty_loss > 0:
            loss_penalty = args.lambda_penalty_loss * loss_func_penalty(output, seq, mask)
            loss += loss_penalty
        else:
            loss_penalty = None

        return loss, loss_penalty

    def loss_func_penalty(output, target_bins, mask, weight=None):

        batch_size, seq_length, num_features_in, num_features_out = output.shape
        bin_width = 1.0 / num_features_out
        if weight is None:
            weight = torch.ones_like(target_bins, dtype=torch.float32, device=target_bins.device)

        # Suppress the probabilites of bins above the previous target bin
        # This is applied to the second satellite and onwards
        prev_target_bins = target_bins[:, 1:-1, 0] # (batch, seq_length-2)
        bin_idx = torch.arange(num_features_out, device=target_bins.device)
        mask_penalty = (bin_idx[None, None, :] > prev_target_bins[:, :, None]).float() # (batch, seq_length-2, num_features_out)
        loss_penalty = ( output[:, 2:, 0, :] * mask_penalty * bin_width ).sum(dim=-1) # (batch, seq_length-2)
        loss_penalty = ( loss_penalty * weight[:, 2:, 0] ).sum() / ( (weight[:, 2:, 0]).sum() + 1e-8 )

        return loss_penalty

    fname_log = "{}/log.txt".format(args.output_dir)

    with open(fname_log, "w") as f:
        f.write(f"# loss loss_val\n")

        num_batches = len(train_dataloader)

        elist = range(args.num_epochs)
        if args.show_pbar:
            elist = tqdm(elist, file=sys.stderr)
            
        for epoch in elist:
            model.train()

            for count, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                loss, loss_penalty = loss_func(batch) #, weight=weight)

                loss.backward()
                optimizer.step()

                model.eval()
                for batch_val in val_dataloader:
                    with torch.no_grad():
                        loss_val, loss_penalty_val = loss_func(batch_val)
                        break # show one batch result only
                model.train()

                epoch_now = epoch + count / num_batches
                
                log = "{:.8f} {:.4f} {:.4f} ".format(epoch_now, loss.item(), loss_val.item())
                if loss_penalty_val is not None:
                    log += "{:.4f} {:.4f} ".format(epoch_now, loss_penalty.item(), loss_penalty_val.item())
                f.write("{}\n".format(log))

            scheduler.step()
            
            # save model
            
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs: 
                fname = "{}/model_ep{:d}.pth".format(args.output_dir, epoch+1)
                torch.save(model.state_dict(), fname)
                if args.show_pbar:
                    tqdm.write("# Model saved to {}".format(fname))
                else:
                    print("# Model saved to {}".format(fname))

                fname = "{}/model.pth".format(args.output_dir)
                torch.save(model.state_dict(), fname)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
