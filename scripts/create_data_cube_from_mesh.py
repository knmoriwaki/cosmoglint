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

#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
import astropy.units as u

from cosmoglint.utils import normalize, namespace_to_dict, get_index_list
from cosmoglint.utils.io_utils import load_mesh_data, save_hdf5_intensity_data, save_hdf5_catalog_data
from cosmoglint.sampling import sample_galaxies_from_mesh_continuous
from cosmoglint.model.transformer import transformer_model


cspeed = 3e10 # [cm/s]
micron = 1e-4 # [cm]

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--verbose", action="store_true", default=False)

    ### I/O parameters
    parser.add_argument("--input_fname", type=str, default=None, help="Input filename")
    parser.add_argument("--output_fname", type=str, default=None, help="Output filename")
    parser.add_argument("--output_catalog_fname", type=str, default=None, help="Output catalog filename")

    parser.add_argument("--global_param_file", type=str, default=None, help="File containing global parameters")
    parser.add_argument("--global_param_id", type=int, default=0, help="Row ID in the global parameter file")

    ### Output format parameters
    parser.add_argument("--npix_to_use", type=int, default=64, help="Npix of input map to be used")
    parser.add_argument("--npix", type=int, default=100, help="Number of pixels in x and y direction")
    parser.add_argument("--npix_z", type=int, default=90, help="Number of pixels in z direction")
    parser.add_argument("--redshift_space", action="store_true", default=False, help="Generate both real and redshift space data")

    parser.add_argument("--intensity_name", type=str, default="SubhaloSFR", help="Name of parameter to be used as intensity")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Galaxies with val > threshold [Msun/yr] will be used")
    parser.add_argument("--catalog_threshold", type=float, default=0, help="Galaxies with val > catalog_threshold [Msun/yr] will be saved in catalog")

    ### Generative model parameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 7th column as intensity.")
    parser.add_argument("--model_label", type=str, default="", help="Model label (e.g., _ep100)")
    parser.add_argument("--prob_threshold", type=float, default=1e-5, help="Below this probability, the galaxy is not generated.")
    parser.add_argument("--monotonicity_start_index", type=int, default=None)
    parser.add_argument("--num_rounds", type=int, default=8, help="Number of rounds of patch generation")

    return parser.parse_args()



def create_data(args):
    
    if args.input_fname is None:
        ValueError("Input filename is not specified. Use --input_fname to specify the input file.")

    if args.model_dir == None:
        raise ValueError("Model directory is not specified. Use --model_dir to specify the model directory.")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print("# Using device: {}".format(device))

    ### Load model
    print("# Load model from {}".format(args.model_dir))
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        opt.norm_param_dict = namespace_to_dict(opt.norm_param_dict)

    model = transformer_model(opt)
    model.load_state_dict(torch.load("{}/model{}.pth".format(args.model_dir, args.model_label), map_location="cpu"))
    model.to(device)
    model.eval()
    print(model)

    if args.verbose:
        print("opt: ", opt)
        print(model)

    if args.batch_size is not None:
        opt.batch_size = args.batch_size

    ### Load input data
    x_in, pixel_size = load_mesh_data(args.input_fname, opt.input_features, norm_param_dict=opt.norm_param_dict)    
    if args.npix_to_use > 0:
        x_in = x_in[-args.npix_to_use:, -args.npix_to_use:, -args.npix_to_use:]
    args.BoxSize = pixel_size * x_in.shape[0]
    with h5py.File(args.input_fname, "r") as f:
        args.redshift = f["Header"].attrs["Redshift"]

    x_in = torch.from_numpy(x_in).float().to(device)

    ### Load global parameters
    if args.global_param_file is not None:
        global_params_all = np.genfromtxt(args.global_param_file, names=True, dtype=None, encoding="utf-8")
        global_params = global_params_all[args.global_param_id]
        global_params = global_params[opt.global_features].to_numpy(dtype=np.float32)
        for i, key in enumerate(opt.global_features):
            global_params[...,i] = normalize(global_params[...,i], key, opt.norm_param_dict)
        global_params = torch.tensor(np.array(global_params), dtype=torch.float32).to(device)
    else:
        global_params = None

    ### Generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    
    stop_criterion = normalize(args.threshold, opt.output_features[0], opt.norm_param_dict) # stop criterion

    generation_kwargs = {
        "global_cond": global_params,
        "prob_threshold": args.prob_threshold,
        "monotonicity_start_index": args.monotonicity_start_index,
        "stop_criterion": stop_criterion,
        "max_ids": None
    }

    generated = sample_galaxies_from_mesh_continuous(
        x_in, 
        model, 
        opt, 
        num_rounds=args.num_rounds, 
        device=device,
        **generation_kwargs
    )

    ### Undo normalization
    for i_param, key in enumerate(opt.output_features):
        if "SubhaloPos" in key:
            generated[:,i_param] *= pixel_size # [kpc/h]
        else:
            generated[:,i_param] = normalize(generated[:,i_param], key, opt.norm_param_dict, inverse=True)
        
    print("# Number of valid galaxies: {:d}".format(len(generated)))
    
    pos_idx = get_index_list(opt.output_features, "SubhaloPos")
    val_idx = opt.output_features.index( args.intensity_name ) 
    
    val = generated[:,val_idx].copy() # (N, )

    ### Save galaxy catalog
    if args.output_catalog_fname is not None:
        print("# Generate catalog of galaxies")
        
        catalog_threshold = max(args.threshold, args.catalog_threshold)
        valid_mask = val > catalog_threshold
        generated_valid = generated[valid_mask]

        save_hdf5_catalog_data(generated_valid, args, opt.output_features, args.output_catalog_fname)
        
    ### Save intensity map
    if args.output_fname is not None:
        print("# Assign galaxies to pixels")
        pos = generated[:,pos_idx].copy() # (N, 3)
        pos_real = pos.copy()

        if "SubhaloVel:2" in opt.output_features:
            iy_vel = opt.output_features.index( "SubhaloVel:2" )
        else:
            iy_vel = None

        if iy_vel is not None:
            H = cosmo.H(args.redshift).to(u.km/u.s/u.Mpc).value #[km/s/Mpc]
            hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
            scale_factor = 1 / (1 + args.redshift)

            vz = generated[:,iy_vel[-1]].copy() # (N, ) [km/s]
            pos[:,2] += vz / scale_factor / H * hlittle
            pos_list = [pos_real, pos]
        else:
            pos_list = [pos_real]

        npix_out = np.array([args.npix, args.npix, args.npix_z], dtype=int)
        dx_pix = args.BoxSize / npix_out # (3,)
        
        def make_intensity_map(pos, flux):
            ix_galaxies = (pos / dx_pix).astype(int) # (num_galaxies_valid, 3)    
            valid_mask = np.all((ix_galaxies >= 0) & (ix_galaxies < npix_out), axis=1)
            ix_valid = ix_galaxies[valid_mask]
            flux_valid = flux[valid_mask]

            intensity = np.zeros(npix_out)
            np.add.at(intensity, (ix_valid[:, 0], ix_valid[:, 1], ix_valid[:, 2]), flux_valid)

            return intensity

        intensities = []
        for pos in pos_list:
            valid_mask = val > args.threshold
            intensity = make_intensity_map(pos, val)
            intensities.append(intensity)
        keys = ["intensity", "intensity_rsd"]

        save_hdf5_intensity_data(intensities, args, keys, args.output_fname)

if __name__ == "__main__":
    args = parse_args()
    create_data(args)
