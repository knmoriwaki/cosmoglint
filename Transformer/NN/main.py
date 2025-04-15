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

from model import MyDataset, my_NN_model

parser = argparse.ArgumentParser()

# base parameters
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--seed", type=int, default=12345)

# training parameters
parser.add_argument("--data_dir", type=str, default="TNG_data")
parser.add_argument("--train_ratio", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--use_sampler", action="store_true")

# model parameters
parser.add_argument("--output_dim", type=int, default=50)
parser.add_argument("--hidden_dim", type=int, default=64) 

args = parser.parse_args()

def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = my_NN_model(args)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def loss_func(output, target):
        # output: (batch, num_features, output_dim)
        # target: (batch, num_features)
            
        target_bins = (target * args.output_dim).long()  # (batch, 2)
        target_bins = torch.clamp(target_bins, min=0, max=args.output_dim - 1)
        x_bin = target_bins[:, 0]  # (batch,)
        y_bin = target_bins[:, 1]  # (batch,)

        log_prob = torch.log(output + 1e-8) 

        loss = -log_prob[torch.arange(output.size(0)), x_bin, y_bin].mean() 
        return loss

    fname_log = f"{args.output_dir}/log.txt"

    with open(fname_log, "w") as f:
        f.write(f"# loss loss_val\n")

    for epoch in range(args.num_epochs):
        model.train()

        teacher_forcing_ratio = 1.
        #teacher_forcing_ratio = 1. if epoch < 10 else 0.
        #teacher_forcing_ratio = 1. - epoch / args.num_epochs

        for x, y in train_dataloader:
            x = x.to(device)  # (batch, num_features_in)   
            y = y.to(device)     # (batch, num_features_out)
            

            output = model(x) # (batch, num_features_out, output_dim)
            
            loss = loss_func(output, y)
            
            model.eval()
            for x_val, y_val in val_dataloader:
                with torch.no_grad():
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    output_val = model(x_val)
                    loss_val = loss_func(output_val, y_val)
                    
                    break # show one batch result only
            model.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            with open(fname_log, "a") as f:
                f.write(f"{loss.item():.4f} {loss_val.item():.4f}\n")
            print(f"{loss.item():.4f} {loss_val.item():.4f}")

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

    args.norm_params = norm_params.tolist()
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")


if __name__ == "__main__":
    main()
    
