import os
import sys
import argparse
import json

from tqdm import tqdm
import yaml

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import random_split

from cosmoglint.datasets import HaloDataset, MeshDataset, MeshCtxDataset
from cosmoglint.utils.io_utils import load_global_params
from cosmoglint.model.transformer import transformer_model

def parse_args():

    parser = argparse.ArgumentParser()

    # base parameters
    parser.add_argument("--gpu_id", type=str, default="0")
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

    return parser.parse_args()

def train_model(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print("# Using device: {}".format(device))

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f) or {}

    for k, v in cfg.items():
        setattr(args, k, v)

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
    args.norm_param_dict = norm_param_dict

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

    print("# Training data: {:d}".format(len(train_dataset)))
    print("# Validation data: {:d}".format(len(val_dataset)))

    ### Save arguments
    fname = "{}/args.json".format(args.output_dir)
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print("# Arguments saved to {}".format(fname))

    ### Training
    def calc_loss(batch, weight=None, device=device):

        import torch.nn.functional as F

        seq = batch["target"].to(device)     # (batch, max_length, num_features_in)
        mask = batch["mask"].to(device)   # (batch, max_length)
        condition = batch["condition"]
        if isinstance(condition, dict):
            condition = {k: v.to(device) for k, v in condition.items()}
        else:
            condition = condition.to(device)
        global_cond = batch["global_cond"].to(device) # (batch, num_features_global)
        
        input_seq = seq[:, :-1]
        target = seq

        output = model(condition, input_seq, global_cond=global_cond) # (batch, max_length, num_features_in, num_features_out)
        #_, output = model.generate(condition, seq=seq, teacher_forcing_ratio=teacher_forcing_ratio) 
        # output: (batch, max_length, num_features_in, num_features_out)

        if weight is None:
            weight = torch.ones_like(target, dtype=torch.float32, device=target.device) # (batch, seq_length)

        weight = mask * weight

        log_prob = torch.log( output + 1e-8 )
        target_bins = (target * args.num_features_out).long() # (batch, seq_length, num_features_in) [0, 1] -> [0, num_features_out-1]
        target_bins = torch.clamp(target_bins, min=0, max=args.num_features_out - 1)

        log_prob_flatten = log_prob.contiguous().view(-1, args.num_features_out) # (batch * seq_length * num_features_in, num_features_out)
        target_bins_flatten = target_bins.contiguous().view(-1) # (batch * seq_length * num_features_in, )
        weight_flatten = weight.contiguous().view(-1) # (batch * seq_length * num_features_in, )

        loss_nll = F.nll_loss(log_prob_flatten, target_bins_flatten, reduction='none') 
        loss = (loss_nll * weight_flatten).sum() / ( (weight_flatten).sum() + 1e-8 )

        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

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
                loss = calc_loss(batch) #, weight=weight)

                loss.backward()
                optimizer.step()

                model.eval()
                for batch_val in val_dataloader:
                    with torch.no_grad():
                        loss_val = calc_loss(batch_val)
                        break # show one batch result only
                model.train()

                epoch_now = epoch + count / num_batches
                
                log = "{:.8f} {:.4f} {:.4f} ".format(epoch_now, loss.item(), loss_val.item())
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
    
