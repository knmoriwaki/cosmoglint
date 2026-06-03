import sys
import os
import numpy as np

import h5py
from collections import defaultdict


def save_hdf5_catalog_data(data, args, output_features, output_fname):
    args_dict = vars(args)
    args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}

    from collections import defaultdict
    groups = defaultdict(list)
    for i, name in enumerate(output_features):
        prefix = name.split(":", 1)[0] 
        groups[prefix].append(i)

    # Save
    with h5py.File(output_fname, 'w') as f:
        for key, idxs in groups.items():
            arr = data[:, idxs]            
            f.create_dataset(key, data=arr, compression="gzip")

        for key, value in args_dict.items():
            f.attrs[key] = value

    print(f"# Catalog saved to {output_fname}")

def save_hdf5_intensity_data(intensity, args, output_features, output_fname):
    args_dict = vars(args)
    args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}

    if not isinstance(intensity, list):
        intensity = [intensity]
    
    with h5py.File(output_fname, 'w') as f:

        for key, d in zip(output_features, intensity):
            f.create_dataset(key, data=d)    

        for key, value in args_dict.items():
            f.attrs[key] = value

    print(f"# Data cube saved as {output_fname}")


def convert_to_log(val, val_min):
    log_val = np.full_like(val, val_min)
    mask = val > 10**val_min
    log_val[mask] = np.log10(val[mask])
    return log_val

def convert_to_log_with_sign(val):
    return np.sign(val) * np.log10(np.abs(val) + 1)

def inverse_convert_to_log_with_sign(val):
    return np.sign(val) * ( 10 ** np.abs( val ) - 1 )

def normalize(x, key, norm_param_dict, inverse=False, convert=True):
    """
    x: array-like
    key: str
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

    x = np.array(x)

    if ":" in key:
        key, idx = key.split(":", 1)
    else:
        key, idx = key, 0

    if "/" in key:
        key = key.split("/")[-1]
        
    if norm_param_dict is not None:
        xmin = norm_param_dict[key]["min"]
        xmax = norm_param_dict[key]["max"] 
        norm = norm_param_dict[key]["norm"]

        if inverse:
            x = x * ( xmax - xmin ) + xmin
            if convert:
                if norm == "log":
                    x = 10 ** x
                elif norm == "log_with_sign":
                    x = inverse_convert_to_log_with_sign(x)
        else:
            if convert:
                if norm == "log":
                    x = convert_to_log(x, xmin)    
                elif norm == "log_with_sign":
                    x = convert_to_log_with_sign(x)
            x = ( x - xmin ) / ( xmax - xmin )

    return x

def load_values(f, key, norm_param_dict=None):
    if key not in f:
        raise ValueError(f"Key '{key}' not found in the file.")

    data = f[key][:]
    if norm_param_dict is None:
        return data
    else:
        return normalize(data, key, norm_param_dict)

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
            data = np.genfromtxt(f, names=True, dtype=None, encoding="utf-8")
            global_params_now = np.vstack([data[name] for name in global_features]).T.astype(np.float32)
            global_params.append(global_params_now)

        global_params = np.vstack(global_params)

        for i, key in enumerate(global_features):
            global_params[...,i] = normalize(global_params[...,i], key, norm_param_dict)        

    return global_params # (ndata, num_features_global)

def load_mesh_data(
        file_path, 
        features,
        norm_param_dict=None
    ):
    """
    Input:
        file_path: Path to the HDF5 file containing the data.
        norm_param_dict: Normalization parameters, if None, normalization is not applied.
    """
        
    print("# Input file (mesh): {}".format(file_path))
    with h5py.File(file_path, "r") as f:
        boxsize = f["Header"].attrs["BoxSize"] # [kpc/h]
        source_list = []
        for feature in features:
            x = load_values(f, feature, norm_param_dict=norm_param_dict) # (npix, npix, npix, C)
            if x.ndim == 3:
                x = x[..., np.newaxis]
            source_list.append(x)

    source = np.concatenate(source_list, axis=-1) # (npix, npix, npix, num_features)
    npix = source.shape[0]   
    pixel_size = boxsize / npix # [kpc/h]

    return source, pixel_size
    
def load_galaxy_data(file_path, features, norm_param_dict):
    key_to_indices = defaultdict(list)

    for feat in features:
        if ":" in feat:
            key, idx = feat.split(":", 1)
        else:
            key, idx = feat, 0
        
        key_to_indices[key].append(int(idx))

    print("# Input file (galaxy): {}".format(file_path))

    with h5py.File(file_path, "r") as f:
        gal_data_list = []
        for key, idxs in key_to_indices.items():
            x = load_values(f, f"Subhalo/{key}", norm_param_dict=norm_param_dict)
            if x.ndim == 1:
                x = x[:, None]
            x = x[:, idxs]
            gal_data_list.append(x)

    gal_data = np.concatenate(gal_data_list, axis=1) # (N, num_features)

    # mask
    mask = (gal_data > 0).all(axis=1)
    gal_data = gal_data[mask]

    return gal_data

 
