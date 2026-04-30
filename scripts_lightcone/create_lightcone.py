import os
import sys
import argparse
import json
import copy
import h5py
import re

from tqdm import tqdm

import numpy as np

import torch

import time

#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
import astropy.units as u

cspeed = 3e10  # [cm/s]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    ### I/O parameters
    parser.add_argument("--input_fname", type=str, default="./Pinocchio/output/pinocchio.r01000.plc.out")
    parser.add_argument("--output_fname", type=str, default="test.h5")

    ### Output format parameters
    parser.add_argument("--redshift_space", action="store_true", default=False, help="Use redshift space")
    parser.add_argument("--gen_both", action="store_true", default=False, help="Generate both real and redshift space data")

    parser.add_argument("--redshift_min", type=float, default=0.0, help="Minimum redshift")
    parser.add_argument("--redshift_max", type=float, default=6.0, help="Maximum redshift")
    parser.add_argument("--dz", type=float, default=0.01, help="Redshift bin size. Not used if gen_catalog is set.")
    parser.add_argument("--use_logz", action="store_true", default=False, help="Use dlogz instead of dz for redshift binning")

    parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Threshold for SFR")

    parser.add_argument("--mass_correction_factor", type=float, default=1.0, help="Mass correction factor")

    parser.add_argument("--gen_catalog", action="store_true", default=False, help="Generate galaxy catalog with SFR > catalog_threshold")
    parser.add_argument("--catalog_threshold", type=float, default=10, help="Threshold for SFR in the catalog")

    parser.add_argument("--side_length", type=float, default=300.0, help="side length in arcsec")
    parser.add_argument("--angular_resolution", type=float, default=30, help="angular resolution in arcsec. Not used if gen_catalog is set.")
    
    ### Generative model parameters
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 4th column as intensity.")
    parser.add_argument("--model_config_file", type=str, default="model_config.json", help="The configuration file for the model")
    parser.add_argument("--param_dir", type=str, default=None, help="The directory of the parameter files")

    return parser.parse_args()

def create_mock(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    print("# redshift: {:.4f} - {:.4f} [GHz]".format(args.redshift_min, args.redshift_max))
    print("# area : {:.4f} arcsec x {:.4f} arcsec".format(args.side_length, args.side_length))
    if not args.gen_catalog:
        print("# dz: {:.4f}".format(args.dz))
        print("# angular resolution : {:.4f} arcsec".format(args.angular_resolution))

    if args.gen_both:
        NotImplementedError("Generating both real and redshift space data is not implemented yet.")

    ### Load data
    mass, pos_x, pos_y, redshift_obs, redshift_real = load_lightcone_data(args.input_fname, cosmo=cosmo)
    mass *= args.mass_correction_factor

    if "pinocchio" in args.input_fname:
        # Pinocchio's lightcone has a circular area
        radius = pos_x.max()
        pos_x += radius / np.sqrt(2)
        pos_y += radius / np.sqrt(2)
        print("# Shift positions -- new max pos: ({:.4f}, {:.4f})".format(pos_x.max(), pos_y.max()))

    if args.redshift_space:
        print("# Using redshift space")
    else:
        print("# Using real space")
        redshift_obs = copy.deepcopy(redshift_real)

    ### Load global parameters
    if args.global_param_file is not None:
        global_params_all = np.genfromtxt(args.global_param_file, names=True, dtype=None, encoding="utf-8")
        global_params = global_params_all[args.global_param_id]
    else:
        global_params = None

    ### Mask out small halos
    mask = (np.log10(mass) > args.logm_min)
    mask = mask & (redshift_obs >= args.redshift_min) & (redshift_obs <= args.redshift_max)
    
    mass = mass[mask]
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    redshift_real = redshift_real[mask]
    redshift_obs = redshift_obs[mask] # Observed redshift if redshift_space is True, otherwise equals to redshift_real

    pos = np.stack([pos_x, pos_y, redshift_obs], axis=1) # (num_halos, 3)
        
    ### Create mock data
    if args.model_dir == None:
        ValueError("Please specify the model directory with --model_dir")
        
    else:
        opt = copy.deepcopy(args)

        generated_all = []
        pos_central_all = [] 
        redshift_central_all = [] 
        flag_central_all = []

        with open(args.model_config_file, "r") as f:
            snapshot_dict_str = json.load(f)
            snapshot_dict = {int(k): v for k, v in snapshot_dict_str.items()}

        print("# Model config:", snapshot_dict)
        redshifts_of_snapshots = np.array([ v[1] for v in snapshot_dict.values() ])    
        bin_edges = (redshifts_of_snapshots[:-1] + redshifts_of_snapshots[1:]) / 2.0
        bin_indices = np.digitize(redshift_real, bin_edges)  

        if args.param_dir is None:
            max_sfr_file_list = [ None for snapshot_number in snapshot_dict ]
        else:
            max_sfr_file_list = ["{}/max_nbin20_{:d}.txt".format(args.param_dir, snapshot_number) for snapshot_number in snapshot_dict]

        time_start = time.time()

        for i, snapshot_number in enumerate(snapshot_dict):
            model_path, redshift_of_snapshot = snapshot_dict[snapshot_number]
            print("# Snapshot number: {:d}, Redshift: {:.2f}".format(snapshot_number, redshift_of_snapshot))
            
            ### Skip if no haloes in this redshift bin
            mask_z = (bin_indices == i)
            if not np.any(mask_z):
                print("# No haloes in redshift bin {:d} (snapshot number {:d}), skipping...".format(i, snapshot_number))
                continue
        
            x_now = mass[mask_z, None] # (num_halos_in_bin, 1)
            pos_now = pos[mask_z] # (num_halos_in_bin, 3)
            redshift_now = redshift_real[mask_z] # (num_halos_in_bin, 1)

            opt.model_dir = "{}/{}".format(args.model_dir, model_path)
            opt.max_sfr_file = max_sfr_file_list[i]

            if "Transformer_NF" in opt.model_dir:
                from cosmoglint.sampling import sample_galaxies_TransNF
                generated, mask = sample_galaxies_TransNF(opt, x_now, global_params=global_params, verbose=False)
            else:
                from cosmoglint.sampling import sample_galaxies
                generated, mask = sample_galaxies(opt, x_now, global_params=global_params, verbose=False)
                
            seq_length = mask.shape[1]
            num_features = generated.shape[-1]

            # Define flag_central
            flag_central = np.zeros_like(mask, dtype=bool)
            flag_central[:, 0] = True

            # Flatten the arrays
            mask = mask.reshape(-1)
            generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)
            pos_central = np.repeat(pos_now[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
            redshift_central = np.repeat(redshift_now[:,None], seq_length, axis=1).reshape(-1) # (num_halos * seq_length)
            flag_central = flag_central.reshape(-1)

            # Apply mask to arrays
            generated = generated[mask] # (num_galaxies_valid, num_features)
            pos_central = pos_central[mask] # (num_galaxies_valid, 3)
            redshift_central = redshift_central[mask] # (num_galaxies_valid, 3)
            flag_central = flag_central[mask] # (num_galaxies_valid, )
            
            # Append
            generated_all.append(generated)
            pos_central_all.append(pos_central)
            redshift_central_all.append(redshift_central)
            flag_central_all.append(flag_central)

        generated_all = np.concatenate(generated_all, axis=0) # (num_galaxies_valid, num_features)
        pos_central_all = np.concatenate(pos_central_all, axis=0)
        redshift_central_all = np.concatenate(redshift_central_all, axis=0) # (num_galaxies_valid,)
        flag_central_all = np.concatenate(flag_central_all, axis=0) # (num_galaxies_valid,)
        
        ### Distribute galaxies in lightcone
        sfr = generated_all[:,0]
        distance = generated_all[:,1]

        num_gal = len(sfr)

        # Determine positions of galaxies
        print("# Generate positions of galaxies")
        _phi = np.random.uniform(0, 2 * np.pi, size=num_gal)
        _cos_theta = np.random.uniform(-1, 1, size=num_gal)
        _sin_theta = np.sqrt(1 - _cos_theta ** 2)
        
        # Convert Mpc to deg
        from cosmoglint.utils.cosmology_utils import cMpc_to_arcsec, dcMpc_to_dz
        distance_arcsec = cMpc_to_arcsec(distance, redshift_central_all, cosmo=cosmo, l_with_hlittle=True)
        distance_z = dcMpc_to_dz(distance, redshift_central_all, cosmo=cosmo, l_with_hlittle=True)

        pos_galaxies = pos_central_all
        pos_galaxies[:,0] += distance_arcsec * _sin_theta * np.cos(_phi)
        pos_galaxies[:,1] += distance_arcsec * _sin_theta * np.sin(_phi)
        pos_galaxies[:,2] += distance_z * _cos_theta
        
        # Add redshift-space distortion
        if args.redshift_space:

            relative_vel_rad = generated_all[:,2]
            relative_vel_tan = generated_all[:,3]
            relative_vel_rad[flag_central_all] = 0 # Set vr to 0 for central galaxies
            alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
            vz_gal = - relative_vel_rad * _cos_theta + relative_vel_tan * _sin_theta * np.cos(alpha)
            
            beta = vz_gal / (cspeed * 100) # [(km/s) / (km/s)]

            redshift_rest = pos_galaxies[:,2]
            pos_galaxies[:,2] = ( 1. + redshift_rest ) * np.sqrt( (1. + beta) / (1. - beta) ) - 1.0

        print(f"# Elapsed time: {time.time() - time_start} sec")

        if args.gen_catalog:

            mask = (sfr > args.catalog_threshold)
            pos_galaxies = pos_galaxies[mask]
            redshift_real = redshift_real[mask]
            sfr = sfr[mask]
            
            with h5py.File(args.output_fname, "w") as f:
                
                args_dict = vars(args)
                args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
                for key, value in args_dict.items():
                    f.attrs[key] = value

                f.create_dataset("Redshifts", data=redshift_real, compression="gzip")
                f.create_dataset("Positions", data=pos_galaxies, compression="gzip")
                f.create_dataset("SFR", data=sfr, compression="gzip")
            
            print("Galaxy catalog saved to {}".format(args.output_fname))

        else:
            ### Initialize the data cube and flist
            Nx = int(args.side_length / args.angular_resolution)
            ix = np.floor(pos_galaxies[:,0] / args.angular_resolution).astype(np.int32)
            iy = np.floor(pos_galaxies[:,1] / args.angular_resolution).astype(np.int32)

            if args.use_logz:
                logz_min_p1 = np.log10(1 + args.redshift_min)
                logz_max_p1 = np.log10(1 + args.redshift_max)
                Nz = int( (logz_max_p1 - logz_min_p1) / args.dz )
                iz = np.floor((np.log10(1 + pos_galaxies[:,2]) - logz_min_p1) / args.dz).astype(np.int32)
            else:
                Nz = int( (args.redshift_max - args.redshift_min) / args.dz )
                iz = np.floor((pos_galaxies[:,2] - args.redshift_min) / args.dz).astype(np.int32)

            indices = np.array([ix, iy, iz]).T # (num_galaxies, 3)

            npix = np.array([Nx, Nx, Nz])
            valid_mask = np.all((indices >= 0) & (indices < npix), axis=1)            

            if np.sum(valid_mask) > 0:
                indices_valid = indices[valid_mask]
                sfr_valid = sfr[valid_mask]
                
                total_intensity = np.zeros((Nx, Nx, Nz), dtype=np.float32)
                np.add.at(total_intensity, (indices_valid[:, 0], indices_valid[:, 1], indices_valid[:, 2]), sfr_valid)

                with h5py.File(args.output_fname, "w") as f:

                    args_dict = vars(args)
                    args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
                    for key, value in args_dict.items():
                        f.attrs[key] = value
                    
                    f.create_dataset("SFR", data=total_intensity, compression="gzip")

                print("SFR map saved to {}".format(args.output_fname))

            else:
                print("No valid galaxies found within the specified bounds. No data saved.") 


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
       
        
if __name__ == "__main__":
    args = parse_args()
    create_mock(args)