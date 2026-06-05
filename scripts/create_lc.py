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

from cosmoglint.utils.io_utils import load_global_params
from cosmoglint.utils.cosmology_utils import ckpc_to_arcsec, dckpc_to_dz
from cosmoglint.utils.misc import get_feature_values, spherical_offsets_and_vz
from cosmoglint.sampling.from_halo import flatten_and_mask_generated

cspeed = 3e10  # [cm/s]

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    ### I/O files
    parser.add_argument("--input_fname", type=str, default="./Pinocchio/output/pinocchio.r01000.plc.out")
    parser.add_argument("--output_fname", type=str, default=None, help="Output filename")
    parser.add_argument("--output_catalog_fname", type=str, default=None, help="Output catalog filename")
    
    parser.add_argument("--global_param_file", type=str, default=None, help="File containing global parameters")
    parser.add_argument("--global_param_id", type=int, default=0, help="Row ID in the global parameter file")

    ### Output format parameters
    parser.add_argument("--redshift_space", action="store_true", default=False, help="Use redshift space")

    parser.add_argument("--redshift_min", type=float, default=0.0, help="Minimum redshift")
    parser.add_argument("--redshift_max", type=float, default=6.0, help="Maximum redshift")
    parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Threshold for SFR")
    parser.add_argument("--mass_correction_factor", type=float, default=1.0, help="Mass correction factor")


    parser.add_argument("--side_length", type=float, default=300.0, help="side length in arcsec")

    ### Output parameters (catalog)
    parser.add_argument("--catalog_threshold", type=float, default=10, help="Threshold for SFR in the catalog")

    ### Output parameters (intensity map)
    parser.add_argument("--line_list", type=str, nargs="+",default=["[CII]"], help="list of line names")

    parser.add_argument("--angular_resolution", type=float, default=30, help="angular resolution in arcsec.")
    parser.add_argument("--fmin", type=float, default=10.0, help="minimum frequency in GHz")
    parser.add_argument("--fmax", type=float, default=100.0, help="maximum frequency in GHz")
    parser.add_argument("--R", type=float, default=100, help="spectral resolution R")
    parser.add_argument("--intensity_unit", type=str, default="Jy/sr", help="Intensity unit to use. Default is Jy/sr.")
    parser.add_argument("--sigma", type=float, default=0.2, help="Log-normal scatter [dex] added to the luminosity–SFR relation.")

    ### Generative model parameters
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model.")
    parser.add_argument("--model_config_file", type=str, default="model_config.json", help="The configuration file for the model")
    parser.add_argument("--param_dir", type=str, default=None, help="The directory of the parameter files")

    return parser.parse_args()

def generate_galaxies_in_multiple_redshifts(
        args, 
        x_in, 
        pos,
        redshift_real
    ):
    
    ### Load global parameters
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    if args.global_param_file is not None:
        global_params = load_global_params(args.global_param_file, opt.global_features)[args.global_param_id] 
    else:
        global_params = None

    ### Reset opt
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

    for i, snapshot_number in enumerate(snapshot_dict):
        model_path, redshift_of_snapshot = snapshot_dict[snapshot_number]
        print("# Snapshot number: {:d}, Redshift: {:.2f}".format(snapshot_number, redshift_of_snapshot))
        
        ### Skip if no haloes in this redshift bin
        mask_z = (bin_indices == i)
        if not np.any(mask_z):
            print("# No haloes in redshift bin {:d} (snapshot number {:d}), skipping...".format(i, snapshot_number))
            continue
    
        x_now = x_in[mask_z, None] # (num_halos_in_bin, 1)
        pos_now = pos[mask_z] # (num_halos_in_bin, 3)
        redshift_now = redshift_real[mask_z] # (num_halos_in_bin, 1)

        opt.model_dir = "{}/{}".format(args.model_dir, model_path)
        opt.max_sfr_file = "{}/max_nbin20_{:d}.txt".format(args.param_dir, snapshot_number) if args.param_dir is not None else None

        if "Transformer_NF" in opt.model_dir:
            from cosmoglint.sampling import sample_galaxies_TransNF
            generated, mask = sample_galaxies_TransNF(opt, x_now, global_params=global_params, verbose=False)
        else:
            from cosmoglint.sampling import sample_galaxies
            generated, mask = sample_galaxies(opt, x_now, global_params=global_params, verbose=False)
            
        # flatten and mask
        out = flatten_and_mask_generated(
            generated,
            mask,
            pos_central=pos_now,
            redshift_central=redshift_now
        )

        # Append
        generated_all.append( out["generated"] )
        pos_central_all.append( out["pos_central"])
        redshift_central_all.append( out["redshift_central"])
        flag_central_all.append( out["flag_central"] )

    generated_all = np.concatenate(generated_all, axis=0) # (num_galaxies_valid, num_features)
    pos_central_all = np.concatenate(pos_central_all, axis=0)
    redshift_central_all = np.concatenate(redshift_central_all, axis=0) # (num_galaxies_valid,)
    flag_central_all = np.concatenate(flag_central_all, axis=0) # (num_galaxies_valid,)

    return generated_all, pos_central_all, redshift_central_all, flag_central_all

def create_lightcone(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    print("# redshift: {:.4f} - {:.4f} [GHz]".format(args.redshift_min, args.redshift_max))
    print("# area : {:.4f} arcsec x {:.4f} arcsec".format(args.side_length, args.side_length))
    print("# angular resolution : {:.4f} arcsec".format(args.angular_resolution))

    ### Load data ###

    mass, pos_x, pos_y, redshift_obs, redshift_real = load_lightcone_data(args.input_fname, cosmo=cosmo)
    mass *= args.mass_correction_factor

    if "pinocchio" in args.input_fname:
        # Pinocchio's lightcone has a circular area
        radius = pos_x.max()
        pos_x += radius / np.sqrt(2)
        pos_y += radius / np.sqrt(2)
        print("# Shift positions -- new max pos: ({:.4f}, {:.4f})".format(pos_x.max(), pos_y.max()))

    if not args.redshift_space:
        redshift_obs = copy.deepcopy(redshift_real)

    mask = (np.log10(mass) + 10 > args.logm_min)
    mask = mask & (redshift_obs >= args.redshift_min) & (redshift_obs <= args.redshift_max)
    
    mass = mass[mask]
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    redshift_real = redshift_real[mask]
    redshift_obs = redshift_obs[mask] 

    pos = np.stack([pos_x, pos_y, redshift_obs], axis=1) # (num_halos, 3)

    ### Generate galaxies ###
    time_start = time.time()

    generated, pos_central, redshift_central, flag_central = generate_galaxies_in_multiple_redshifts(
        args,
        x_in = mass,
        pos = pos,
        redshift_real = redshift_real
    )

    ### Add sphereical offsets and redshift-space distortion ###
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    pos_galaxies = add_spherical_offset_and_rsd(
        generated = generated,
        pos_central = pos_central,
        redshift_central = redshift_central,
        flag_central = flag_central,
        output_features = opt.output_features, 
        redshift_space = args.redshift_space
    )

    print(f"# Elapsed time: {time.time() - time_start} sec")

    ### Save
    if args.output_fname is not None:
        
        ### Generate line intensity map ###

        from line_intensity_map import create_and_save_line_intensity_map
        i_sfr = opt.output_features.index("SubhaloSFR")
        log_sfr = np.log10( generated[:,i_sfr] )
        create_and_save_line_intensity_map(
            pos_x=pos_galaxies[:,0],
            pos_y=pos_galaxies[:,1],
            z_obs=pos_galaxies[:,2],
            z_real=redshift_central,
            log_sfr=log_sfr,
            fmin=args.fmin,
            fmax=args.fmax,
            R=args.R,
            side_length=args.side_length,
            angular_resolution=args.angular_resolution,
            line_list=args.line_list,
            intensity_unit=args.intensity_unit,
            sigma=args.sigma,
            args=args
        )

    if args.output_catalog_fname is not None:

        ### Generate catalog ###
        mask = (generated[:,0] > args.catalog_threshold)
        pos_galaxies = pos_galaxies[mask]
        redshift = redshift_central[mask]
        luminosity_list = [ luminosity[mask] for luminosity in luminosity_list ]
        
        with h5py.File(args.output_fname, "w") as f:
            
            args_dict = vars(args)
            args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
            for key, value in args_dict.items():
                f.attrs[key] = value

            f.create_dataset("Redshifts", data=redshift, compression="gzip")
            f.create_dataset("Positions", data=pos_galaxies, compression="gzip")
            
            for iparam, key in enumerate(opt.output_features):
                f.create_dataset(key, data=generated[:,iparam], compression="gzip")
        
        print("Galaxy catalog saved to {}".format(args.output_fname))


def add_spherical_offset_and_rsd(
    generated, 
    pos_central, 
    redshift_central, 
    flag_central, 
    output_features = [],
    redshift_space = False
):
    
    distance = get_feature_values(generated, output_features, "SubhaloDist")
    vr = get_feature_values(generated, output_features, "SubhaloVrad")
    vt = get_feature_values(generated, output_features, "SubhaloVtan")

    distance_arcsec = ckpc_to_arcsec(distance, redshift_central, cosmo=cosmo, l_with_hlittle=True)
    distance_z = dckpc_to_dz(distance, redshift_central, cosmo=cosmo, l_with_hlittle=True)

    offset, vz = spherical_offsets_and_vz(
        distance_arcsec, 
        distance_z = distance_z, 
        vr = vr, 
        vt = vt, 
        flag_central=flag_central
    )

    pos_galaxies = pos_central + offset

    if redshift_space:
        beta = vz / (cspeed * 100) # [(km/s) / (km/s)]
        redshift_rest = pos_galaxies[:,2]
        pos_galaxies[:,2] = ( 1. + redshift_rest ) * np.sqrt( (1. + beta) / (1. - beta) ) - 1.0

    return pos_galaxies


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
    create_lightcone(args)