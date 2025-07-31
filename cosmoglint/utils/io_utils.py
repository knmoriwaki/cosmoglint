import os

import random
import numpy as np

import h5py

import torch

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


def load_halo_data(file_path, max_length=10, norm_params=None, ndata=None, use_dist=False, use_vel=False, sort=True):
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
        return log_val, mask

    def convert_to_log_with_sign(val):
        return np.sign(val) * np.log10(np.abs(val) + 1), None
        
    def load_values(f, key, min_val, max_val):
        try:
            data = f[key][:]
            if key == "HaloMass":
                data *= 1e10 / f.attrs["Hubble"]
                        
            if key == "SubgroupVrad":
                data, mask = convert_to_log_with_sign(data)
            else:
                data, mask = convert_to_log(data, min_val)
            
            data = ( data - min_val ) / ( max_val - min_val )
            
            return data, mask
        except KeyError:
            print(f"Key '{key}' not found in the file.")
            return None, None

    print(f"# Loading halo data from {file_path}")
    with h5py.File(file_path, "r") as f:
        mass, mask = load_values(f, "HaloMass", xmin[0], xmax[0])
        sfr, _ = load_values(f, "SubgroupSFR", xmin[1], xmax[1])
        num_params = 1
        if use_dist:
            dist, _ = load_values(f, "SubgroupDist", xmin[2], xmax[2])
            num_params += 1
        if use_vel:
            vrad, _ = load_values(f, "SubgroupVrad", xmin[3], xmax[3])
            vtan, _ = load_values(f, "SubgroupVtan", xmin[4], xmax[4])
            num_params += 2

        num_subgroups = f["NumSubgroups"][:]
        offset = f["Offset"][:]
        
        for j in range(len(mass)):
            if not mask[j]:
                continue

            start = offset[j]
            end = start + num_subgroups[j]
            
            if num_subgroups[j] == 0:
                y_j = np.zeros((1, num_params)) # handle empty subgroups
            elif use_vel:
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
    def __init__(self, path, max_length=10, norm_params=None, ndata=None, use_dist=False, use_vel=False, sort=True):
        
        if not isinstance(path, list):
            path = [path]

        self.x = torch.empty((0, 1))
        self.y = []

        for p in path:
            x_tmp, y_tmp = load_halo_data(p, max_length=max_length, norm_params=norm_params, ndata=ndata, use_dist=use_dist, use_vel=use_vel, sort=sort)
            self.x = torch.cat([self.x, x_tmp], dim=0)
            self.y = self.y + y_tmp

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
    
def load_lightcone_data(input_fname, cosmo):
    print(f"# Load {input_fname}")

    if "pinocchio" in input_fname: 
        if "old_version" in input_fname:
            M, theta, phi, _, redshift_obs, redshift_real = load_old_plc(input_fname)
            logm = np.log10(M)
        else:
            import cosmoglint.utils.ReadPinocchio5 as rp
            myplc = rp.plc(input_fname)
            
            logm = np.log10( myplc.data["Mass"] )
            theta = myplc.data["theta"] # [arcsec]
            phi = myplc.data["phi"]

            redshift_obs = myplc.data["obsz"]
            redshift_real = myplc.data["truez"]

        import astropy.units as u
        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
        logm -= np.log10(hlittle) # [Msun]

        theta = ( 90. - theta ) * 3600 # [arcsec]
        pos_x = theta * np.cos( phi * np.pi / 180. ) # [arcsec] 
        pos_y = theta * np.sin( phi * np.pi / 180. ) # [arcsec]

        theta_max = np.max(theta)
        pos_x += theta_max / 1.5
        pos_y += theta_max / 1.5
        
        print("# Minimum log mass in catalog: {:.5f}".format(np.min(logm)))
        print("# Maximum pos_x: {:.3f} arcsec".format(np.max(pos_x)))
        print("# Maximum pos_y: {:.3f} arcsec".format(np.max(pos_y)))
        print("# Redshift: {:.3f} - {:.3f}".format(np.min(redshift_real), np.max(redshift_real)))
        print("# Number of halos: {}".format(len(logm)))

    else:
        raise ValueError("Unknown input file format")
    
    return logm, pos_x, pos_y, redshift_obs, redshift_real


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
