import os
import sys
import argparse
import json

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split

from cosmoglint.utils import MyDataset, load_global_params
from cosmoglint.model import transformer_nf_model, my_stop_predictor, calculate_transformer_nf_loss

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=12345)
    
    # dataset parameters
    parser.add_argument("--data_path", type=str, nargs='+', default=["data.h5"])
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
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--exclude_ratio", type=float, default=0.0, help="Exclude halos in the corner of a size (exclude_ratio * BoxSize)^3")

    # model parameters
    parser.add_argument("--model_name", type=str, default="transformer1")

    parser.add_argument("--d_model", type=int, default=128, help="hidden dimension of transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")

    parser.add_argument("--base_dist", type=str, default="normal", help="base distribution")
    parser.add_argument("--num_context", type=int, default=4, help="number of context features")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension of flow") 
    parser.add_argument("--num_flows", type=int, default=4, help="number of flows")

    parser.add_argument("--lambda_stop", type=float, default=1, help="weight for stop prediction loss")
    parser.add_argument("--hidden_dim_stop", type=int, default=64, help="hidden dimension of stop predictor")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")

    return parser.parse_args()


def train_model(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ### Load model
    args.num_features_cond = len(args.input_features)
    args.num_features_in = len(args.output_features)
    args.num_features_global = 0 if args.global_features is None else len(args.global_features)

    model, flow = transformer_nf_model(args)
            
    model.to(device)
    flow.to(device)
    
    if args.verbose:
        print(model)
        print(flow)

    ### Load data
    with open(args.norm_param_file, "r") as f:
        norm_param_dict = json.load(f)

    global_params = load_global_params(args.global_param_file, args.global_features, norm_param_dict=norm_param_dict)
    
    if "*" in args.data_path[0] and args.indices is not None:
        # Currently only support one data path with *
        if len(args.data_path) > 1:
            raise ValueError("When data_path contains *, only one data path is allowed.")
        
        indices = args.indices.split("-")
        istart = int(indices[0])
        iend = int(indices[1])
        print(f"# Using data files from {istart} to {iend}")
        args.data_path = [ args.data_path[0].replace("*", str(i)) for i in range(istart, iend+1) ]
        
        if global_params is not None:
            global_params = global_params[istart:iend+1, :]

    dataset =  MyDataset(args.data_path, args.input_features, args.output_features, global_params=global_params, norm_param_dict=norm_param_dict, max_length=args.max_length, exclude_ratio=args.exclude_ratio)
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
    
    print(f"# Training data: {len(train_dataset)}")
    print(f"# Validation data: {len(val_dataset)}")
        
    ### Save arguments
    args.norm_param_dict = norm_param_dict
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")

    ### Training
    params = list(model.parameters()) + list(flow.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    fname_log = f"{args.output_dir}/log.txt"
    with open(fname_log, "w") as f:
        f.write(f"#epoch loss loss_val\n")

        num_batches = len(train_dataloader)
        for epoch in tqdm(range(args.num_epochs)):
            model.train()

            for count, (condition, seq, global_cond, mask) in enumerate(train_dataloader):

                model.eval() # val evaluation first for ActNorm in flow
                for condition_val, seq_val, global_cond_val, mask_val, in val_dataloader:
                    with torch.no_grad():
                        loss_val = calculate_transformer_nf_loss(model, flow, condition_val, seq_val, mask_val)
                        break # show one batch result only
                model.train()
                
                optimizer.zero_grad()
                
                loss = calculate_transformer_nf_loss(model, flow, condition, seq, mask)

                loss.backward()
                optimizer.step()
            
                epoch_now = epoch + count / num_batches
                
                f.write(f"{epoch_now:.4f} {loss.item():.4f} {loss_val.item():.4f}\n")

            scheduler.step()

            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs: 
                fname = "{}/model_ep{:d}.pth".format(args.output_dir, epoch+1)
                torch.save(model.state_dict(), fname)
                tqdm.write("# Model saved to {}".format(fname))

                fname = "{}/model_ep{:d}.pth".format(args.output_dir, epoch+1)
                torch.save(flow.state_dict(), fname)
                tqdm.write("# Model saved to {}".format(fname))

                fname = "{}/model.pth".format(args.output_dir)
                torch.save(model.state_dict(), fname)
                fname = "{}/flow.pth".format(args.output_dir)
                torch.save(flow.state_dict(), fname)

                
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
