import os
import sys
import argparse
import json

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split

from cosmoglint.utils import MyDataset
from cosmoglint.model import transformer_nf_model, my_stop_predictor, calculate_transformer_nf_loss

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--input_features", type=str, nargs='+', default=["HaloMass"])
    parser.add_argument("--output_features", type=str, nargs='+', default=["SubgroupSFR", "SubgroupDist", "SubgroupVrad", "SubgroupVtan"])
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--norm_param_file", type=str, default="./norm_params.json")

    # training parameters
    parser.add_argument("--data_path", type=str, nargs='+', default=["data.h5"])
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
    args.num_condition = len(input_features)
    args.num_features = len(output_features)

    model, flow = transformer_nf_model(args)
    
    if args.load_epoch > 0:
        model.load_state_dict(torch.load(f"{args.output_dir}/model_ep{args.load_epoch}.pth", map_location="cpu"))
        flow.load_state_dict(torch.load(f"{args.output_dir}/flow_ep{args.load_epoch}.pth", map_location="cpu"))
        
    model.to(device)
    flow.to(device)
    
    if args.verbose:
        print(model)
        print(flow)

    ### Load data
    with open(args.norm_param_file, "r") as f:
        norm_param_dict = json.load(f)
    dataset = MyDataset(args.data_path, input_features=args.input_features, output_features=args.output_features, max_length=args.max_length, norm_param_dict=norm_param_dict)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if args.use_sampler:
        def get_sampler(x, nbins=20, xmin=0, xmax=1):
            bins = torch.linspace(xmin, xmax, steps=nbins+1)
            bin_indices = torch.bucketize(x, bins)
            counts = torch.bincount(bin_indices)
            weights = 1. / counts[bin_indices]
            weights = torch.clamp(weights, min=0.02, max=1) # avoid too small weights
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
        
    ### Save arguments
    args.norm_param_dict = norm_param_dict
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")

    ### Training
    params = list(model.parameters()) + list(flow.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=args.load_epoch-1)

    fname_log = f"{args.output_dir}/log.txt"
    mode = "a" if args.load_epoch > 0 else "w"
    with open(fname_log, mode) as f:
        f.write(f"#epoch loss loss_val\n")

        num_batches = len(train_dataloader)
        for epoch in tqdm(range(args.num_epochs)):
            model.train()

            count = 0
            for condition, seq, mask in train_dataloader:

                model.eval() # val evaluation first for ActNorm in flow
                for condition_val, seq_val, mask_val, in val_dataloader:
                    with torch.no_grad():
                        loss_val = calculate_transformer_nf_loss(model, flow, condition_val, seq_val, mask_val)
                        break # show one batch result only

                model.train()
                optimizer.zero_grad()
                
                loss = calculate_transformer_nf_loss(model, flow, condition, seq, mask)

                loss.backward()
                optimizer.step()
            
                epoch_now = epoch + count / num_batches
                if args.load_epoch > 0:
                    epoch_now += args.load_epoch
                
                f.write(f"{epoch_now:.4f} {loss.item():.4f} {loss_val.item():.4f}\n")
            
                count += 1

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
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
