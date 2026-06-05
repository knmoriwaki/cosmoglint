import sys
import os
import numpy as np

import h5py
from collections import defaultdict

# ============================================================
# Save functions
# ============================================================

def save_hdf5_data(
    data_list, 
    key_list, 
    fname,
    args = None,
):

    groups = defaultdict(list)
    for i, name in enumerate(key_list):
        groups[name].append(i)

    # Save
    with h5py.File(fname, 'w') as f:
        for key, idxs in groups.items():
            arr = data_list[:, idxs]            
            f.create_dataset(key, data=arr, compression="gzip")

        if args is not None:
            args_dict = vars(args)
            args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}

            for key, value in args_dict.items():
                f.attrs[key] = value

    print(f"# Catalog saved to {fname}")

# ============================================================
# Normalization functions
# ============================================================

def normalize(
    x, 
    key, 
    norm_param_dict, 
    inverse=False, 
    convert=True
):
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

def convert_to_log(val, val_min):
    log_val = np.full_like(val, val_min)
    mask = val > 10**val_min
    log_val[mask] = np.log10(val[mask])
    return log_val

def convert_to_log_with_sign(val):
    return np.sign(val) * np.log10(np.abs(val) + 1)

def inverse_convert_to_log_with_sign(val):
    return np.sign(val) * ( 10 ** np.abs( val ) - 1 )

# ============================================================
# Load values from hdf5 file
# ============================================================

def load_values(
    f, 
    key, 
    norm_param_dict=None
):

    if key not in f:
        raise ValueError(f"Key '{key}' not found in the file.")

    data = f[key][:]
    if norm_param_dict is None:
        return data
    else:
        return normalize(data, key, norm_param_dict)
    
def load_header_values(
    f, 
    key, 
    norm_param_dict=None
):
    
    if "Header" in f and key in f["Header"].attrs:
        data = f["Header"].attrs[key]

        if key is "Redshift":
            data += 0.1 * np.random.normal(0, 0.1) # Add scatter to learn intermediate redshifts
            
        if norm_param_dict is None:
            return data
        else:
            return normalize(data, key, norm_param_dict)                
    else:
        return None
    
# ============================================================
# Load global parameters from ascii file
# ============================================================

def load_global_params(
    global_param_file, 
    global_features, 
    norm_param_dict=None
):
    """
    Load global parameters from ascii file 

    Input:
        global_param_file: Path to the ASCII file containing the data.
        global_features: List of feature names (e.g., ["Omega0"])
        norm_param_dict: Normalization parameters, if None, normalization is not applied.

    Return:
        global_params: np.ndarray, shape (ndata, len(global_features))
    """

    if global_features is None:
        return None

    if global_param_file is None:
        return None
    
    if not isinstance(global_param_file, list):
        global_param_file = [global_param_file]

    global_params = []
    for f in global_param_file:
        data = np.genfromtxt(f, names=True, dtype=None, encoding="utf-8")
        data = np.atleast_1d(data)

        global_params_now = []
        for name in global_features:
            if name in data.dtype.names:
                values = data[name]
            else:
                values = np.full(len(data), np.nan) # This not-found value will be replaced by the parameter obtained in data file. If not, ValueError will be raised.

            global_params_now.append(values)

        global_params_now = np.vstack(global_params_now).T.astype(np.float32)
        global_params.append(global_params_now)

    global_params = np.vstack(global_params)

    for i, key in enumerate(global_features):
        values = global_params[...,i]
        valid = ~np.isnan(values)
        global_params[valid,i] = normalize(values[valid], key, norm_param_dict)        

    return global_params # (ndata, num_features_global)

# ============================================================
# Load mesh data from hdf5 file
# ============================================================

def load_mesh_data(
    file_path, 
    features,
    norm_param_dict=None
):
    """
    Load mesh data from hdf5 file

    Input:
        file_path: Path to the HDF5 file containing the data.
        features: List of feature names (e.g., ["dm_density"])
        norm_param_dict: Normalization parameters, if None, normalization is not applied.

    Return:
        source: np.ndarray, shape (npix, npix, npix, len(features))
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
    
# ============================================================
# Load galaxy data from hdf5 file
# ============================================================

def load_galaxy_data(
    file_path, 
    features, 
    global_features=None, 
    norm_param_dict=None
):
    """
    Input:
        file_path: Path to the HDF5 file containing the data.
        features: List of feature names (e.g., ["dm_density"])
        norm_param_dict: Normalization parameters, if None, normalization is not applied.
    """

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

        if global_features is not None:
            g_list = []
            for feature in global_features:
                g = load_header_values(f, feature, norm_param_dict=norm_param_dict)
                g_list.append(g)
        else:
            g_list = None

    gal_data = np.concatenate(gal_data_list, axis=1) # (N, num_features)

    # mask
    mask = (gal_data > 0).all(axis=1)
    gal_data = gal_data[mask]

    return gal_data, g_list

 
