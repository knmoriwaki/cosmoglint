import os

import random
import numpy as np

import h5py

import torch

from torch.utils.data import Dataset

def my_save_model(model, fname):
    torch.save(model.state_dict(), fname)
    print(f"# Model saved to {fname}")

def my_load_model(model, fname):
    model.load_state_dict(torch.load(fname))
    print(f"# Model loaded from {fname}")

def load_halo_data(data_dir, max_length=10, norm_params=None, ndata=None, use_dist=False, use_vel=False, sort=True):
    if norm_params is None:
        xmin, xmax = np.zeros(5), np.ones(5)
    else:
        xmin = norm_params[:,0]
        xmax = norm_params[:,1]

    y_list = []

    def convert_to_log(val, val_min):
        log_val = np.full_like(val, val_min)
        mask = val > 10**val_min
        log_val[mask] = np.log10(val[mask])
        return log_val

    def convert_to_log_with_sign(val):
        return np.sign(val) * np.log10(np.abs(val) + 1)
    
    def normalize(val, val_min, val_max):
        return (val - val_min) / (val_max - val_min)
    
    def load_values(f, key, min_val, max_val):
        try:
            data = f[key][:]
            if key == "HaloMass":
                data *= 1e10 / f.attrs["Hubble"]
                        
            if key == "SubgroupVrad":
                data = convert_to_log_with_sign(data)
            else:
                data = convert_to_log(data, min_val)
            
            data = normalize(data, min_val, max_val)

            mask = data > 0
            
            return data, mask
        except KeyError:
            print(f"Key '{key}' not found in the file.")
            return None, None

    snapshot_number = 38
    file_path = os.path.join(data_dir, f"TNG300-1_{snapshot_number}.h5")
    with h5py.File(file_path, "r") as f:
        mass, mask = load_values(f, "HaloMass", xmin[0], xmax[0])
        sfr, _ = load_values(f, "SubgroupSFR", xmin[1], xmax[1])
        if use_dist:
            dist, _ = load_values(f, "SubgroupDist", xmin[2], xmax[2])
        if use_vel:
            vrad, _ = load_values(f, "SubgroupVrad", xmin[3], xmax[3])
            vtan, _ = load_values(f, "SubgroupVtan", xmin[4], xmax[4])

        num_subgroups = f["NumSubgroups"][:]
        offset = f["Offset"][:]
        
        for j in range(len(mass)):
            if not mask[j]:
                continue

            start = offset[j]
            end = start + num_subgroups[j]
            if use_vel:
                y_j = np.stack([sfr[start:end], dist[start:end], vrad[start:end], vtan[start:end]], axis=1) # (num_subgroups, 2)
            elif use_dist:
                y_j = np.stack([sfr[start:end], dist[start:end]], axis=1)
            else:
                y_j = sfr[start:end, None] # (num_subgroups, 1)

            if sort:
                # Sort by sfr
                sorted_indices = [0] + sorted(range(1, len(y_j)), key=lambda k: y_j[k,0], reverse=True)
                y_j = y_j[sorted_indices]

            y_j = y_j[:max_length] # truncate
            y_j = torch.tensor(y_j, dtype=torch.float32)
            y_list.append(y_j)
            
    mass = mass[mask]
    mass = torch.tensor(mass, dtype=torch.float32)
    mass = mass.unsqueeze(1) # (N, 1)

    if ndata is not None:
        mass = mass[:ndata]
        y_list = y_list[:ndata]

    return mass, y_list

class MyDataset(Dataset):
    def __init__(self, data_dir, max_length=10, norm_params=None, ndata=None, use_dist=False, use_vel=False, sort=True):
        
        self.x, self.y = load_halo_data(data_dir, max_length=max_length, norm_params=norm_params, ndata=ndata, use_dist=use_dist, use_vel=use_vel, sort=sort)

        _, num_params = (self.y[0]).shape

        #self.y_padded = torch.zeros(len(self.x), max_length, num_params)
        self.y_padded = torch.zeros(len(self.x), max_length, num_params)
        self.mask = torch.zeros(len(self.x), max_length, num_params, dtype=torch.bool)
        
        for i, y_i in enumerate(self.y):
            length = len(y_i)
            self.y_padded[i, :length, :] = y_i
            self.mask[i, :length+1, :] = True # use the last + 1 value to learn when to stop
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_padded[idx], self.mask[idx]
    