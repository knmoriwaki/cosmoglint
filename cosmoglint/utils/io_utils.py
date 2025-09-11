import os
from argparse import Namespace
import random
import numpy as np
import pandas as pd

import h5py

import torch

from tqdm import tqdm

from torch.utils.data import Dataset


def my_save_model(model, fname):
    torch.save(model.state_dict(), fname)
    print(f"# Model saved to {fname}")

def save_catalog_data(pos_list, value, args, output_fname):
    if not isinstance(pos_list, list):
        pos_list = [pos_list]

    with open(output_fname, 'w') as f:
        for i, v in enumerate(value):
            f.write(f"{pos_list[0][i, 0]} {pos_list[0][i, 1]} ")
            for pos in pos_list:
                f.write(f"{pos[i, 2]} ")

            f.write(f"{v}\n")

    print(f"# Catalog saved to {output_fname}")

def save_intensity_data(intensity, args, output_fname):
    args_dict = vars(args)
    args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}

    if not isinstance(intensity, list):
        intensity = [intensity]
    
    with h5py.File(output_fname, 'w') as f:
        for i, d in enumerate(intensity):
            f.create_dataset(f'intensity{i}', data=d)    
        for key, value in args_dict.items():
            f.attrs[key] = value
    print(f"# Data cube saved as {output_fname}")

def namespace_to_dict(ns):
    if isinstance(ns, Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns

def convert_to_log(val, val_min):
    log_val = np.full_like(val, val_min)
    mask = val > 10**val_min
    log_val[mask] = np.log10(val[mask])
    return log_val

def convert_to_log_with_sign(val):
    return np.sign(val) * np.log10(np.abs(val) + 1)

def inverse_convert_to_log_with_sign(val):
    return np.sign(val) * ( 10 ** np.abs( val ) - 1 )

def normalize(x, keys, norm_param_dict, inverse=False, convert=True):
    """
    x: array-like, shape (N,) or (N, num_features)
    keys: list of str
    norm_param_dict: dict
        e.g., {
            "GroupMass": {"min": 10, "max": 15, "norm": "log"},
            ...
        }
    inverse: bool
        If True, perform inverse normalization.
    convert: bool
        If True, convert to/from log scale based on norm_param_dict.
    """

    get_item = False
    squeeze = False

    x = np.array(x)

    if x.ndim == 0:
        x = x[None, None]
        get_item = True
    
    if x.ndim == 1:
        if len(keys) == 1:
            x = x[:, None]
        else:
            x = x[None, :]
        squeeze = True


    if not isinstance(keys, list):
        keys = [keys]

    if (len(keys) == 1) and (x.shape[1] != 1):
        x = x[..., None]
        squeeze = True

    if x.shape[-1] != len(keys):
        raise ValueError("Input x has shape {}, but expected {} features".format(x.shape, len(keys)))
        
    if norm_param_dict is not None:
        xmin = np.array([(norm_param_dict[key])["min"] for key in keys])
        xmax = np.array([norm_param_dict[key]["max"] for key in keys])
        norm_mode = [ norm_param_dict[key]["norm"] for key in keys ]

        if inverse:
            x = x * ( xmax - xmin ) + xmin
            if convert:
                for i, mode in enumerate(norm_mode):
                    if mode == "log":
                        x[...,i] = 10 ** x[..., i]
                    elif mode == "log_with_sign":
                        x[...,i] = inverse_convert_to_log_with_sign(x[...,i])
        else:
            if convert:
                for i, mode in enumerate(norm_mode):
                    if mode == "log":
                        x[...,i] = convert_to_log(x[...,i], xmin[i])    
                    elif mode == "log_with_sign":
                        x[...,i] = convert_to_log_with_sign(x[...,i])
            x = ( x - xmin ) / ( xmax - xmin )
            
    if squeeze:
        x = np.squeeze(x)

    if get_item:
        x = x.item()

    return x

def load_global_params(global_param_file, global_features, norm_param_dict=None):

    if global_features is None:
        global_params = None

    else:
        if global_param_file is None:
            raise ValueError("global_param_file must be specified when global_features is provided.")
        
        if not isinstance(global_param_file, list):
            global_param_file = [global_param_file]

        global_params = []
        for f in global_param_file:
            global_params_pd = pd.read_csv(f, sep=r"\s+")
            global_params_now = global_params_pd[global_features].to_numpy(dtype=np.float32)
            
            global_params.append(global_params_now)

        global_params = np.vstack(global_params)

        global_params = normalize(global_params, global_features, norm_param_dict)        

    return global_params # (ndata, num_features_global)
    
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
        
    def load_values(f, key):
        if key not in f:
            raise ValueError(f"Key '{key}' not found in the file.")

        data = f[key][:]
        return data

    num_features_in = len(input_features)
    num_features_out = len(output_features)

    with h5py.File(file_path, "r") as f:

        # Load input features
        source_list = []
        for feature in input_features:
            x = load_values(f, feature)
            source_list.append(x)

        source = np.stack(source_list, axis=1)  # (N, num_features_in)
        source = normalize(source, input_features, norm_param_dict)

        mask = np.ones(len(source), dtype=bool)
        for i in range(num_features_in):
            mask = mask & ( source[:,i] > 0 )
            
        if exclude_ratio > 0:
            boxsize = f.attrs["BoxSize"] # [kpc/h]
            halo_pos = f["GroupPos"][:]  # [kpc/h]
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
            y = load_values(f, feature)
            target_list.append(y)

        target = np.stack(target_list, axis=1)  # (N, num_features_out)
        target = normalize(target, output_features, norm_param_dict)

        num_subgroups = f["GroupNsubs"][:]

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

class MyDataset(Dataset):
    def __init__(
            self, 
            path,  
            input_features,
            output_features, 
            global_params=None,
            norm_param_dict=None, 
            max_length=10, 
            sort=True,
            ndata=None, 
            exclude_ratio=0.0,
            use_excluded_region=False,
        ):
            
        if not isinstance(path, list):
            path = [path]

        if global_params is not None:
            if len(global_params) != len(path):
                raise ValueError("The number of global parameter sets must match the number of data files")

        self.x = torch.empty((0, len(input_features)), dtype=torch.float32)
        self.y = []
        self.g = []

        if len(path) < 20:
            verbose = True 
        else:
            verbose = False
            print("# Loading halo data from {} to {} ({} files)".format(path[0], path[-1], len(path)))

        for i, p in tqdm(enumerate(path)):
            if verbose:
                print(f"# Loading halo data from {p}")
    
            x_tmp, y_tmp = load_halo_data(p, input_features, output_features, norm_param_dict=norm_param_dict, max_length=max_length, sort=sort, ndata=ndata, exclude_ratio=exclude_ratio, use_excluded_region=use_excluded_region)
            self.x = torch.cat([self.x, x_tmp], dim=0)
            self.y = self.y + y_tmp

            if global_params is not None:
                global_param = global_params[i]
                g_tmp = np.repeat(global_param[None, :], len(x_tmp), axis=0) # (Nhalo, num_features_global)
            else:
                g_tmp = np.zeros((len(x_tmp), 1)) # dummy (Nhalo, 1)
            
            self.g.append( g_tmp )

        if len(self.x) == 0:
            raise ValueError("No halo is found.")

        self.g = np.vstack(self.g) 
        self.g = torch.tensor(self.g, dtype=torch.float32)
        
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
        return self.x[idx], self.y_padded[idx], self.g[idx], self.mask[idx]
    
def load_lightcone_data(input_fname, cosmo):
    print(f"# Load {input_fname}")

    if "pinocchio" in input_fname: 
        if "old_version" in input_fname:
            M, theta, phi, _, redshift_obs, redshift_real = load_old_plc(input_fname)
            mass = M
        else:
            import cosmoglint.utils.ReadPinocchio5 as rp
            myplc = rp.plc(input_fname)
            
            mass = myplc.data["Mass"] 
            theta = myplc.data["theta"] # [arcsec]
            phi = myplc.data["phi"]

            redshift_obs = myplc.data["obsz"]
            redshift_real = myplc.data["truez"]

        import astropy.units as u
        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
        mass /= hlittle # [Msun]

        theta = ( 90. - theta ) * 3600 # [arcsec]
        pos_x = theta * np.cos( phi * np.pi / 180. ) # [arcsec] 
        pos_y = theta * np.sin( phi * np.pi / 180. ) # [arcsec]

        
        print("# Minimum log mass in catalog: {:.5f}".format(np.min(np.log10(mass))))
        print("# Maximum pos_x: {:.3f} arcsec".format(np.max(pos_x)))
        print("# Maximum pos_y: {:.3f} arcsec".format(np.max(pos_y)))
        print("# Redshift: {:.3f} - {:.3f}".format(np.min(redshift_real), np.max(redshift_real)))
        print("# Number of halos: {}".format(len(mass)))

    else:
        raise ValueError("Unknown input file format")
    
    return mass, pos_x, pos_y, redshift_obs, redshift_real


def load_old_plc(filename):
    import struct

    plc_struct_format = "<Q d ddd ddd ddddd"  # Q=uint64, d=double, little-endian
    plc_size = struct.calcsize(plc_struct_format)

    M_list = []
    th_list = []
    ph_list = []
    vl_list = []
    zo_list = []
    z_list = []
    with open(filename, "rb") as f:
        while True:
            dummy_bytes = f.read(4)
            if not dummy_bytes:
                break  # EOF
            dummy = struct.unpack("<i", dummy_bytes)[0]

            plc_bytes = f.read(dummy)
            if len(plc_bytes) != dummy:
                break  # 不完全な読み込み

            data = struct.unpack(plc_struct_format, plc_bytes)
            (
                id, z, x1, x2, x3, v1, v2, v3,
                M, th, ph, vl, zo
            ) = data

            dummy2_bytes = f.read(4)
            dummy2 = struct.unpack("<i", dummy2_bytes)[0]

            M_list.append(M)
            th_list.append(th)
            ph_list.append(ph)
            vl_list.append(vl)
            zo_list.append(zo)
            z_list.append(z)
    
    return np.array(M_list), np.array(th_list), np.array(ph_list), np.array(vl_list), np.array(zo_list), np.array(z_list)
