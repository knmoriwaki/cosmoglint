import os
import sys
import argparse
import json
import copy
import h5py
import re

from tqdm import tqdm

import numpy as np

import numpy as np

import torch

#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
import astropy.units as u

cspeed = 3e10 # [cm/s]
micron = 1e-4 # [cm]
GHz = 1e9
Jy = 1.0e-23        # jansky (erg/s/cm2/Hz)
arcsec = 4.848136811094e-6 # [rad] ... arcmin / 60 //

from cosmoglint.utils.lightcone_utils import cMpc_to_arcsec, dcMpc_to_dz, load_lightcone_data, generate_galaxy_in_lightcone

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

    ### Generate mock data with frequency bins if --gen_mock is set otherwise galaxy catalog with SFR > catalog_threshold is created
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
    print("# dz: {:.4f}".format(args.dz))
    print("# area : {:.4f} arcsec x {:.4f} arcsec".format(args.side_length, args.side_length))
    print("# angular resolution : {:.4f} arcsec".format(args.angular_resolution))

    if args.gen_both:
        NotImplementedError("Generating both real and redshift space data is not implemented yet.")

    ### Load data
    logm, pos_x, pos_y, redshift_obs, redshift_real = load_lightcone_data(args.input_fname, cosmo=cosmo)
    
    logm += np.log10(args.mass_correction_factor)

    if args.redshift_space:
        print("# Using redshift space")
    else:
        print("# Using real space")
        redshift_obs = copy.deepcopy(redshift_real)

    ### Mask out small halos
    mask = (logm > args.logm_min)
    mask = mask & (redshift_obs >= args.redshift_min) & (redshift_obs <= args.redshift_max)
    
    logm = logm[mask]
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    redshift_real = redshift_real[mask]
    redshift_obs = redshift_obs[mask] # Observed redshift if redshift_space is True, otherwise equals to redshift_real

    pos = np.stack([pos_x, pos_y, redshift_obs], axis=1) # (num_halos, 3)
        
    ### Create mock data
    if args.model_dir == None:
        ValueError("Please specify the model directory with --model_dir")
        
    else:
        if "Transformer_NF" in args.model_dir:
            ValueError("Transformer_NF model is not supported yet. Please use a different model.")
        else:
            generated, pos_central, redshift_real_central, flag_central = generate_galaxy_in_lightcone(args, logm, pos, redshift_real)

        sfr = generated[:,0]
        distance = generated[:,1]

        num_gal = len(sfr)

        redshift_real = redshift_real_central

        ### Determine positions of galaxies
        print("# Generate positions of galaxies")
        _phi = np.random.uniform(0, 2 * np.pi, size=num_gal)
        _cos_theta = np.random.uniform(-1, 1, size=num_gal)
        _sin_theta = np.sqrt(1 - _cos_theta ** 2)
        
        # Convert Mpc to deg
        distance_arcsec = cMpc_to_arcsec(distance, redshift_real, cosmo=cosmo, l_with_hlittle=True)
        distance_z = dcMpc_to_dz(distance, redshift_real, cosmo=cosmo, l_with_hlittle=True)

        pos_galaxies = pos_central
        pos_galaxies[:,0] += distance_arcsec * _sin_theta * np.cos(_phi)
        pos_galaxies[:,1] += distance_arcsec * _sin_theta * np.sin(_phi)
        pos_galaxies[:,2] += distance_z * _cos_theta
        
        if args.redshift_space:

            relative_vel_rad = generated[:,2]
            relative_vel_tan = generated[:,3]
            relative_vel_rad[flag_central] = 0 # Set vr to 0 for central galaxies
            alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
            vz_gal = - relative_vel_rad * _cos_theta + relative_vel_tan * _sin_theta * np.cos(alpha)
            
            beta = vz_gal / (cspeed * 100) # [(km/s) / (km/s)]

            redshift_rest = pos_galaxies[:,2]
            pos_galaxies[:,2] = ( 1. + redshift_rest ) * np.sqrt( (1. + beta) / (1. - beta) ) - 1.0
            
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
        
if __name__ == "__main__":
    args = parse_args()
    create_mock(args)