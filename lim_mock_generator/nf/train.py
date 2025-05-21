import os
import sys
import argparse
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split
from torch.distributions import Beta
import torch.nn.functional as F

from model import MyDataset, my_flow_model

def parse_args():
    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=12345)

    # training parameters
    parser.add_argument("--data_dir", type=str, default="TNG_data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--use_sampler", action="store_true")

    # model parameters
    parser.add_argument("--hidden_dim", type=int, default=64) 
    parser.add_argument("--num_layers", type=int, default=5)

    return parser.parse_args()

def train_model(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load data
    norm_params = np.loadtxt("../norm_params.txt")

    dataset = MyDataset(args.data_dir, norm_params=norm_params)
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
    
    print(f"# Training data: {len(train_dataset)}")
    print(f"# Validation data: {len(val_dataset)}")

    args.num_features_in = dataset.x.shape[1]
    args.num_features_out = dataset.y.shape[1]

    ### Load model 
    model = my_flow_model(args)
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

    fname_log = f"{args.output_dir}/log.txt"

    with open(fname_log, "w") as f:
        f.write(f"# loss loss_val\n")

    for epoch in range(args.num_epochs):
        model.train()

        for x, y in train_dataloader:

            model.eval() # evaluate val loss first for ActNorm
            for x_val, y_val in val_dataloader:
                with torch.no_grad():
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    loss_val = - model.log_prob(y_val, x_val).mean()
                    break # show one batch result only
            model.train()


            x = x.to(device)  # (batch, num_features_in)   
            y = y.to(device)     # (batch, num_features_out)
            
            loss = - model.log_prob(y, context=x).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            with open(fname_log, "a") as f:
                f.write(f"{loss.item():.4f} {loss_val.item():.4f}\n")
            print(f"{loss.item():.4f} {loss_val.item():.4f}")

        scheduler.step()

    # save model
    fname = f"{args.output_dir}/model.pth"
    torch.save(model.state_dict(), fname)
    print(f"# Model saved to {fname}")

    """
    # show some results for validation data
    fname_log = f"{args.output_dir}/val.txt"
    with open(fname_log, "w") as f:
        with torch.no_grad():
            context_val, seq_val, mask_val = next(iter(val_dataloader))
            output_val = model(context_val.to(device))
            for i in range(args.batch_size):
                f.write(f"{context_val[i].item()} {seq_val[i,0].item()} {output_val[i,0].item()}\n")
    """


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
