import sys
import os

import random
import numpy as np

import h5py

import torch

from tqdm import tqdm

from torch.utils.data import Dataset

from cosmoglint.utils.io_utils import load_values
    
def load_halo_data(
        file_path, 
        input_features,
        output_features,
        norm_param_dict=None, 
        max_length=10, 
        sort=True,
        ndata=None,
        exclude_ratio=0.0, 
        use_excluded_region=False,
    ):
        
    num_features_in = len(input_features)
    num_features_out = len(output_features)

    with h5py.File(file_path, "r") as f:

        # Load input features
        source_list = []
        for feature in input_features:
            x = load_values(f, f"Group/{feature}", norm_param_dict=norm_param_dict)
            source_list.append(x)

        source = np.stack(source_list, axis=1)  # (N, num_features_in)

        mask = np.ones(len(source), dtype=bool)
        for i in range(num_features_in):
            mask = mask & ( source[:,i] > 0 )
            
        if exclude_ratio > 0:
            boxsize = f["Header"].attrs["BoxSize"] # [kpc/h]
            halo_pos = f["Group/GroupPos"][:]  # [kpc/h]
            mask_exclude = (halo_pos[:,0] > boxsize * (1.-exclude_ratio)) \
                        & (halo_pos[:,1] > boxsize * (1.-exclude_ratio)) \
                        & (halo_pos[:,2] > boxsize * (1.-exclude_ratio))

            if use_excluded_region:
                print("# Using excluded region of size ({} * BoxSize)^3".format(exclude_ratio))
                mask = mask & mask_exclude
            else:
                print("# Exclude halos in the corner of size ({} * BoxSize)^3".format(exclude_ratio))
                print("# The excluded region is {:.2f} % of the entire volume".format(100.0 * (exclude_ratio**3)))
                mask = mask & (~mask_exclude)
            
        if mask.sum() == 0:
            print("# No halo is found in {}".format(file_path))
            return torch.empty((0, num_features_in), dtype=torch.float32), []

        # Load output features
        target_list = []
        for feature in output_features:
            y = load_values(f, f"Subhalo/{feature}", norm_param_dict=norm_param_dict)
            target_list.append(y)

        target = np.stack(target_list, axis=1)  # (N, num_features_out)

        num_subgroups = f["Group/GroupNsubs"][:]

        offset = 0
        y_list = []
        for j in range(len(source)):
            start = offset
            end = start + num_subgroups[j]
            offset = end

            if not mask[j]:                
                continue

            if num_subgroups[j] == 0:
                y_j = np.zeros((1, num_features_out)) # handle empty subgroups
            else:
                y_j = target[start:end, :]

            if sort:
                sorted_indices = [0] + sorted(range(1, len(y_j)), key=lambda k: y_j[k,0], reverse=True)
                y_j = y_j[sorted_indices]

            y_j = y_j[:max_length] # truncate
            y_j = torch.tensor(y_j, dtype=torch.float32)
            y_list.append(y_j)
            
    x = source[mask]
    x = torch.tensor(x, dtype=torch.float32)
    
    if ndata is not None:
        x = x[:ndata]
        y_list = y_list[:ndata]

    return x, y_list

class HaloDataset(Dataset):
    def __init__(
            self,
            args,  
            global_params=None,
            sort=True,
            exclude_ratio=0.0,
            use_excluded_region=False,
            show_pbar=True,
        ):
            
        if not isinstance(args.data_path, list):
            args.data_path = [args.data_path]

        if global_params is not None:
            if len(global_params) != len(args.data_path):
                raise ValueError("The number of global parameter sets ({:d}) must match the number of data files ({:d})".format(len(global_params), len(args.data_path)))

        x = []
        self.y = []
        self.g = []

        plist = args.data_path
        if len(plist) < 20:
            verbose = True 
        else:
            verbose = False
            if show_pbar:
                plist = tqdm(plist, file=sys.stderr)
            print("# Loading halo data from {} to {} ({} files)".format(args.data_path[0], args.data_path[-1], len(args.data_path)))

        for i, p in enumerate(plist):
            if verbose:
                print(f"# Loading halo data from {p}")
    
            x_tmp, y_tmp = load_halo_data(p, args.input_features, args.output_features, norm_param_dict=args.norm_param_dict, max_length=args.max_length, sort=sort, ndata=args.ndata, exclude_ratio=exclude_ratio, use_excluded_region=use_excluded_region)
            x.append(x_tmp) 
            self.y = self.y + y_tmp

            if global_params is not None:
                global_param = global_params[i]
                g_tmp = np.repeat(global_param[None, :], len(x_tmp), axis=0) # (Nhalo, num_features_global)
            else:
                g_tmp = np.zeros((len(x_tmp), 1)) # dummy (Nhalo, 1)
            
            self.g.append( g_tmp )

        self.x = torch.cat(x, dim=0)

        if len(self.x) == 0:
            raise ValueError("No halo is found.")

        self.g = np.vstack(self.g) 
        self.g = torch.tensor(self.g, dtype=torch.float32)
        
        _, num_params = (self.y[0]).shape

        self.y_padded = torch.zeros(len(self.x), args.max_length, num_params)
        self.mask = torch.zeros(len(self.x), args.max_length, num_params, dtype=torch.bool)
        
        for i, y_i in enumerate(self.y):
            length = len(y_i)
            self.y_padded[i, :length, :] = y_i[:args.max_length]
            self.mask[i, :length+1, :] = True # use the last + 1 value to learn when to stop
        
        if args.use_flat_representation:
            self.y_padded = self.y_padded.reshape(len(self.y_padded), -1, 1) # (Nhalo, max_length * output_features, 1)
            self.mask = self.mask.reshape(len(self.mask), -1, 1) # (Nhalo, max_length * output_features, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        out = {
            "condition": self.x[idx],
            "global_cond": self.g[idx],
            "target": self.y_padded[idx],
            "mask": self.mask[idx]
        }
        return out
    