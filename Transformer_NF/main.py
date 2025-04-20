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

from utils import MyDataset
from model import my_model, my_flow_model, output_log_prob

parser = argparse.ArgumentParser()

# base parameters
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--max_length", type=int, default=30)
parser.add_argument("--use_dist", action="store_true")
parser.add_argument("--use_vel", action="store_true")
parser.add_argument("--seed", type=int, default=12345)

# training parameters
parser.add_argument("--data_dir", type=str, default="TNG_data")
parser.add_argument("--train_ratio", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--use_sampler", action="store_true")
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--load_epoch", type=int, default=0, help="load model from checkpoint")

# model parameters
parser.add_argument("--model_name", type=str, default="transformer1")

parser.add_argument("--d_model", type=int, default=128, help="hidden dimension of transformer")
parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")

parser.add_argument("--base_dist", type=str, default="gaussian", help="base distribution")
parser.add_argument("--num_context", type=int, default=4, help="number of context features")
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension of flow") 
parser.add_argument("--num_flows", type=int, default=4, help="number of flows")


args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ### Load data
    norm_params = np.loadtxt("./norm_params.txt")
    dataset = MyDataset(args.data_dir, max_length=args.max_length, norm_params=norm_params, use_dist=args.use_dist, use_vel=args.use_vel)
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

    args.num_condition = 1
    _, args.num_features = train_dataset[0][1].shape
    
    print(f"# Training data: {len(train_dataset)}")
    print(f"# Validation data: {len(val_dataset)}")

    ### Load model
    model = my_model(args)
    flow = my_flow_model(args)

    if args.load_epoch > 0:
        my_load_model(model, f"{args.output_dir}/model_ep{args.load_epoch}.pth")
        my_load_model(flow, f"{args.output_dir}/flow_ep{args.load_epoch}.pth")

    model.to(device)
    print(model)
    flow.to(device)
    print(model)

    ### Save arguments
    args.norm_params = norm_params.tolist()
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")

    ### Training
    params = list(model.parameters()) + list(flow.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5, last_epoch=args.load_epoch-1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=args.load_epoch-1)

    fname_log = f"{args.output_dir}/log.txt"
    mode = "a" if args.load_epoch > 0 else "w"
    with open(fname_log, mode) as f:
        f.write(f"#epoch loss loss_val\n")

    num_batches = len(train_dataloader)
    #for epoch in range(args.num_epochs):
    for epoch in tqdm(range(args.num_epochs)):
        model.train()

        count = 0
        for condition, seq, mask in train_dataloader:

            model.eval() # val evaluation first for ActNorm in flow
            for condition_val, seq_val, mask_val in val_dataloader:
                with torch.no_grad():
                    log_prob_val = output_log_prob(model, flow, condition_val, seq_val, mask_val)
                    loss_val = - log_prob_val.mean() / args.num_features
                    break # show one batch result only

            model.train()
            optimizer.zero_grad()
            
            log_prob = output_log_prob(model, flow, condition, seq, mask)
            loss = - log_prob.mean() / args.num_features     

            loss.backward()
            optimizer.step()
        
            epoch_now = epoch + count / num_batches
            if args.load_epoch > 0:
                epoch_now += args.load_epoch
            with open(fname_log, "a") as f:
                f.write(f"{epoch_now:.4f} {loss.item():.4f} {loss_val.item():.4f}\n")
            
            count += 1

        #tqdm.write(f"{epoch_now:.4f} {loss.item():.4f} {loss_val.item():.4f}")
        scheduler.step()


        if epoch + 1 == args.num_epochs or (epoch + 1) % args.save_freq == 0:
            my_save_model(model, f"{args.output_dir}/model.pth")
            my_save_model(flow, f"{args.output_dir}/flow.pth")

            epoch_now = epoch + 1
            if args.load_epoch > 0:
                epoch_now += args.load_epoch
            my_save_model(model, f"{args.output_dir}/model_ep{epoch_now}.pth")
            my_save_model(flow, f"{args.output_dir}/flow_ep{epoch_now}.pth")
    
if __name__ == "__main__":
    main()
    
