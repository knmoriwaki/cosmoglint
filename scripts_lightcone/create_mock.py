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

from line_model import calc_line_luminosity, line_dict
from cosmoglint.utils.cosmology_utils import z_to_log_lumi_dis
from cosmoglint.utils.generation_utils import populate_galaxies_in_lightcone
from cosmoglint.utils.io_utils import load_lightcone_data

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    ### I/O parameters
    parser.add_argument("--input_fname", type=str, default="./Pinocchio/output/pinocchio.r01000.plc.out")
    parser.add_argument("--output_fname", type=str, default="test.h5")

    ### line model parameters
    parser.add_argument("--sigma", type=float, default=0.2, help="The scatter in log [dex] added to luminosity - SFR relation")

    ### Output format parameters
    parser.add_argument("--redshift_space", action="store_true", default=False, help="Use redshift space")
    parser.add_argument("--gen_both", action="store_true", default=False, help="Generate both real and redshift space data")

    parser.add_argument("--redshift_min", type=float, default=0.0, help="Minimum redshift")
    parser.add_argument("--redshift_max", type=float, default=6.0, help="Maximum redshift")

    parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Threshold for SFR")

    parser.add_argument("--mass_correction_factor", type=float, default=1.0, help="Mass correction factor")

    parser.add_argument("--gen_catalog", action="store_true", default=False, help="Generate galaxy catalog with SFR > catalog_threshold")
    parser.add_argument("--catalog_threshold", type=float, default=10, help="Threshold for SFR in the catalog")

    ### Generate mock data with frequency bins if --gen_mock is set otherwise galaxy catalog with SFR > catalog_threshold is created
    parser.add_argument("--side_length", type=float, default=300.0, help="side length in arcsec")
    parser.add_argument("--angular_resolution", type=float, default=30, help="angular resolution in arcsec")
    parser.add_argument("--fmin", type=float, default=10.0, help="minimum frequency in GHz")
    parser.add_argument("--fmax", type=float, default=100.0, help="maximum frequency in GHz")
    parser.add_argument("--R", type=float, default=100, help="spectral resolution R")

    ### Generative model parameters
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 4th column as intensity.")
    parser.add_argument("--model_config_file", type=str, default="model_config.json", help="The configuration file for the model. ")
    parser.add_argument("--param_dir", type=str, default=None, help="The directory of the parameter files")

    return parser.parse_args()

def create_mock(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    print("# frequency: {:.4f} - {:.4f} [GHz]".format(args.fmin, args.fmax))
    print("# spectral resolution R: {:.4f}".format(args.R))
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
            ValueError("Transformer_NF model is not supported yet. Please use the old model.")
        else:
            sfr, pos_galaxies, redshift_real = populate_galaxies_in_lightcone(args, logm, pos, redshift_real, cosmo=cosmo)

        log_sfr = np.log10( sfr )
        log_lumi_dis = z_to_log_lumi_dis(redshift_real, cosmo) # [cm]

        if args.gen_catalog:
            NotImplementedError("Generating galaxy catalog is not implemented yet.")
        else:
            ### Initialize the data cube and flist
            flist = []
            dflist = []

            fnow = args.fmin * GHz
            while fnow <= args.fmax * GHz:
                flist.append(fnow)
                dflist.append(fnow / args.R)
                fnow += fnow / args.R
            flist = np.array(flist, dtype=np.float32)
            dflist = np.array(dflist, dtype=np.float32)

            Nx = int(args.side_length / args.angular_resolution)
            Nz = len(flist) - 1

            npix = np.array([Nx, Nx, Nz])
            total_intensity = np.zeros((Nx, Nx, Nz), dtype=np.float32)

            ix = np.floor(pos_galaxies[:,0] / args.angular_resolution).astype(np.int32)
            iy = np.floor(pos_galaxies[:,1] / args.angular_resolution).astype(np.int32)

            with h5py.File(args.output_fname, "w") as f:

                ### Save metadata
                args_dict = vars(args)
                args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
                for key, value in args_dict.items():
                    f.attrs[key] = value

                f.create_dataset("frequency", data=flist, compression="gzip")

                ### Save intensities 

                for line_name in list(line_dict.keys()):
                    freq_obs = line_dict[line_name][0] / ( 1. + pos_galaxies[:,2] )

                    iz = np.searchsorted(flist, freq_obs, side="right") - 1

                    indices = np.array([ix, iy, iz]).T # (num_galaxies, 3)

                    valid_mask = np.all((indices >= 0) & (indices < npix), axis=1)
                    
                    intensity_line = np.zeros((Nx, Nx, Nz), dtype=np.float32)

                    if np.sum(valid_mask) > 0:
                        indices_valid = indices[valid_mask]
                        log_sfr_valid = log_sfr[valid_mask]
                        z_valid = redshift_real[valid_mask]
                        log_lumi_dis_valid = log_lumi_dis[valid_mask]

                        log_lumi_valid = calc_line_luminosity(args, z_valid, log_sfr_valid, line_name)
                        flux_valid = 10 ** ( log_lumi_valid - 2 * log_lumi_dis_valid ) / ( 4. * np.pi ) # [erg/s/cm2]
                        
                        print("# Found {} valid galaxies for {}; Total flux {:.3e}".format(np.sum(valid_mask), line_name, np.sum(flux_valid)))

                        intensity_valid = flux_valid / dflist[indices_valid[:, 2]] / Jy / ( args.angular_resolution * arcsec )**2 # [Jy/sr]
                            
                        np.add.at(intensity_line, (indices_valid[:, 0], indices_valid[:, 1], indices_valid[:, 2]), intensity_valid)

                        total_intensity += intensity_line

                        f.create_dataset("intensity_{}".format(line_name), data=intensity_line, compression="gzip")

                f.create_dataset("total_intensity", data=total_intensity, compression="gzip")

            print("Intensity map saved to {}".format(args.output_fname))
        
if __name__ == "__main__":
    args = parse_args()
    create_mock(args)