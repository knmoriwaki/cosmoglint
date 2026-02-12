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
            x = load_values(f, key, norm_param_dict=norm_param_dict)
            if x.ndim == 1:
                x = x[:, None]
            x = x[:, idxs]
            gal_data_list.append(x)

    gal_data = np.concatenate(gal_data_list, axis=1) # (N, num_features)

    # mask
    mask = (gal_data > 0).all(axis=1)
    gal_data = gal_data[mask]

    return gal_data

 
def load_lightcone_data(input_fname, cosmo):
    print(f"# Load {input_fname}")

    if "pinocchio" in input_fname: 
        if "old_version" in input_fname:
            M, theta, phi, _, redshift_obs, redshift_real = load_old_plc(input_fname)
            mass = M
        else:
            import ReadPinocchio5 as rp
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
        print("# Maximum pos: ({:.3f}, {:.3f}) arcsec".format(np.max(pos_x), np.max(pos_y)))
        print("# Minimum pos: ({:.3f}, {:.3f}) arcsec".format(np.min(pos_x), np.min(pos_y)))
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
                break  

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
