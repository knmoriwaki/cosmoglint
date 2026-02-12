import os
import sys
import argparse
import json

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split

from cosmoglint.datasets import HaloDataset, MeshDataset, MeshCtxDataset
from cosmoglint.utils.io_utils import load_global_params
from cosmoglint.model.transformer_nf import transformer_nf_model, my_stop_predictor, calculate_transformer_nf_loss

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--show_pbar", action=argparse.BooleanOptionalAction, default=True)

    # dataset and model parameters
    parser.add_argument("--config_file", type=str, default="config.yaml")

    # training parameters
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--exclude_ratio", type=float, default=0.0, help="Exclude halos in the corner of a size (exclude_ratio * BoxSize)^3")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--sampler_xmin", type=float, default=0)
    parser.add_argument("--sampler_xmax", type=float, default=1)
    parser.add_argument("--sampler_weight_min", type=float, default=1, help="Minimum weight for the sampler, set to 1 to disable sampling")    

    parser.add_argument("--save_freq", type=int, default=100)
    
    parser.add_argument("--lambda_stop", type=float, default=1, help="weight for stop prediction loss")
    parser.add_argument("--hidden_dim_stop", type=int, default=64, help="hidden dimension of stop predictor")
    
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
    
    print(model)
    print(flow)

    ### Load data
    with open(args.norm_param_file, "r") as f:
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

    if args.model_name == "mesh_conditioned_transformer": 
        dataset_class = MeshDataset
    elif args.model_name == "mesh_sequence_conditioned_transformer":
        dataset_class = MeshCtxDataset
    else:
        dataset_class = HaloDataset
    dataset = dataset_class(args, global_params=global_params, exclude_ratio=args.exclude_ratio, show_pbar=args.show_pbar)    
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if args.sampler_weight_min < 1:
        from cosmoglint.utils import get_sampler
        x = train_dataset.dataset.x[train_dataset.indices]
        x = x.mean(dim=tuple(range(1, x.ndim)))
        sampler = get_sampler(x, xmin=args.sampler_xmin, xmax=args.sampler_xmax, weight_min=args.sampler_weight_min)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler) 

        x = val_dataset.dataset.x[val_dataset.indices]
        x = x.mean(dim=tuple(range(1, x.ndim)))
        sampler = get_sampler(x, xmin=args.sampler_xmin, xmax=args.sampler_xmax, weight_min=args.sampler_weight_min)
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
        elist = range(args.num_epochs)
        if args.show_pbar:
            elist = tqdm(elist, file=sys.stderr)
        def my_print(log):
            if args.show_pbar:
                tqdm.write(log)
            else:
                print(log)
        for epoch in elist:
            model.train()

            for count, batch in enumerate(train_dataloader):

                model.eval() # val evaluation first for ActNorm in flow
                for batch_val, in val_dataloader:
                    with torch.no_grad():
                        loss_val = calculate_transformer_nf_loss(model, flow, batch_val)
                        break # show one batch result only
                model.train()
                
                optimizer.zero_grad()
                
                loss = calculate_transformer_nf_loss(model, flow, batch)

                loss.backward()
                optimizer.step()
            
                epoch_now = epoch + count / num_batches
                
                f.write(f"{epoch_now:.4f} {loss.item():.4f} {loss_val.item():.4f}\n")

            scheduler.step()

            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs: 
                fname = "{}/model_ep{:d}.pth".format(args.output_dir, epoch+1)
                torch.save(model.state_dict(), fname)
                my_print("# Model saved to {}".format(fname))

                fname = "{}/model_ep{:d}.pth".format(args.output_dir, epoch+1)
                torch.save(flow.state_dict(), fname)
                my_print("# Model saved to {}".format(fname))

                fname = "{}/model.pth".format(args.output_dir)
                torch.save(model.state_dict(), fname)
                fname = "{}/flow.pth".format(args.output_dir)
                torch.save(flow.state_dict(), fname)

                
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
    
