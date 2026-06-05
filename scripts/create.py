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

from cosmoglint.utils.io_utils import save_hdf5_data, load_global_params
from cosmoglint.utils.misc import get_feature_values, spherical_offsets_and_vz
from cosmoglint.sampling.from_halo import flatten_and_mask_generated

cspeed = 3e10 # [cm/s]
micron = 1e-4 # [cm]

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    ### I/O parameters
    parser.add_argument("--input_fname", type=str, default="group.txt", help="Input filename")
    parser.add_argument("--output_fname", type=str, default=None, help="Output filename")
    parser.add_argument("--output_catalog_fname", type=str, default=None, help="Output catalog filename")

    parser.add_argument("--global_param_file", type=str, default=None, help="File containing global parameters")
    parser.add_argument("--global_param_id", type=int, default=0, help="Row ID in the global parameter file")

    parser.add_argument("--boxsize", type=float, default=100.0, help="Box size of data")
    parser.add_argument("--boxsize_to_use", type=float, default=None, help="Box size to be used")

    ### Output format parameters
    parser.add_argument("--npix", type=int, default=100, help="Number of pixels in x and y direction")
    parser.add_argument("--npix_z", type=int, default=90, help="Number of pixels in z direction")

    parser.add_argument("--redshift_space", action="store_true", default=False, help="Create data in both real and redshift space")
    
    parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass [Msun] to be used")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Galaxies with SFR > threshold [Msun/yr] will be used")

    parser.add_argument("--catalog_threshold", type=float, default=10, help="Threshold for SFR in the catalog in [Msun/yr]")

    parser.add_argument("--mass_correction_factor", type=float, default=1.0, help="Mass correction factor; the halo mass is multiplied by this factor before generating galaxies.")

    ### Generative model parameters
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 7th column as intensity.")
    parser.add_argument("--model_label", type=str, default="", help="Model label (e.g., _ep100)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size. If not given, the size used in training wil be used.")

    parser.add_argument("--max_sfr_file", type=str, default=None, help="File containing maximum IDs for SFR.")
    parser.add_argument("--monotonicity_start_index", type=int, default=1)
    parser.add_argument("--prob_threshold", type=float, default=1e-5, help="Below this probability, the galaxy is not generated.")

    return parser.parse_args()


def create_data(args):
    if (args.output_fname is None) and (args.output_catalog_fname is None):
        raise ValueError("Please set at least one of the output_fname or output_catalog_fname.")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    npix = np.array([args.npix, args.npix, args.npix_z])
    if args.boxsize_to_use is None:
        args.boxsize_to_use = args.boxsize
    dx_pix = args.boxsize_to_use / npix

    ### Load data
    redshift, halo_data = get_z_m_p_v_s(args.input_fname)
    halo_data[:,0] *= args.mass_correction_factor

    print("# Minimum log mass in catalog [Msun]: {:.5f}".format(np.min(np.log10(halo_data[:,0])+10)))
    print("# Maximum log mass in catalog [Msun]: {:.5f}".format(np.max(np.log10(halo_data[:,0])+10)))
    print("# Use halos with log mass [Msun] > {}".format(args.logm_min))
    mask = (np.log10(halo_data[:,0]) + 10 > args.logm_min)

    ### Mask data    
    if args.boxsize_to_use < args.boxsize:
        print("# Use a volume at the last corner -- new boxsize: {:.3f}".format(args.boxsize_to_use))
    xmin = args.boxsize - args.boxsize_to_use
    mask = mask & (halo_data[:,1:4] > xmin).all(axis=-1)
    halo_data = halo_data[mask]

    cond = halo_data[:, 0:1] # (nhalo, 1)
    pos = halo_data[:,1:4] - xmin
    vel = halo_data[:,4:7] if args.redshift_space else None

    print(f"# Redshift: {redshift}")

    if args.model_dir is not None:
        device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        ### Load global parameters
        with open("{}/args.json".format(args.model_dir), "r") as f:
            opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
            
        if args.global_param_file is not None:
            global_params = load_global_params(args.global_param_file, opt.global_features)[args.global_param_id] 
        else:
            global_params = None

        ### Generate galaxies
        if "transformer_nf" in args.model_dir:
            from cosmoglint.sampling import sample_galaxies_TransNF
            generated, mask = sample_galaxies_TransNF(
                model_dir = args.model_dir,
                x_in = cond, 
                global_params = global_params, 
                batch_size = args.batch_size,
                model_label = args.model_label,
                threshold = args.threshold,
                device = device
                )
        else:
            from cosmoglint.sampling import sample_galaxies
            generated, mask = sample_galaxies(
                model_dir = args.model_dir,
                x_in = cond, 
                global_params = global_params, 
                batch_size = args.batch_size,
                model_label = args.model_label,
                threshold = args.threshold,
                max_sfr_file = args.max_sfr_file,
                monotonicity_start_index = args.monotonicity_start_index,
                prob_threshold = args.prob_threshold,
                device = device
                )

        ### flatten and mask
        out = flatten_and_mask_generated(generated, mask, pos_central=pos, vel_central=vel)

        generated = out["generated"]
        pos_central = out["pos_central"]
        flag_central = out["flag_central"]
        vel_central = out["vel_central"] if args.redshift_space else None

        ### Add spherical offset and redshift-space distortion
        with open("{}/args.json".format(args.model_dir), "r") as f:
            opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        pos_list, vz = add_spherical_offset_and_rsd(
            generated = generated,
            pos_central = pos_central,
            vel_central = vel_central,
            flag_central = flag_central,
            redshift = redshift,
            output_features = opt.output_features,
            redshift_space = args.redshift_space
        )

        ### Save
        sfr = generated[:,0]

        if args.output_fname is not None:
            print("# Assign galaxies to pixels")
            
            intensities = []
            for pos in pos_list:
                intensity = make_intensity_map(pos, sfr, npix, dx_pix)
                intensities.append(intensity)

            keys = ["intensity", "intensity_rsd"]
            save_hdf5_data(
                data_list = intensities, 
                key_list = keys, 
                fname = args.output_fname, 
                args=args
            )

        if args.output_catalog_fname is not None:
            print("# Generate catalog of galaxies")
            pos = pos_list[0]
            data_list = [pos[:,0], pos[:,1], pos[:,2], sfr]
            output_features = ["SubhaloPos"] * 3 + ["SubhaloSFR"]
            
            if args.redshift_space:
                data_list.append( vel_central[:,2] + vz )
                output_features.append( "SubhaloVelZ" )
                
            mask = sfr > args.catalog_threshold

            save_hdf5_data(
                data_list = data_list, 
                key_list = output_features, 
                fname = args.output_catalog_fname, 
                args=args
            )

    else:
        print("# Use original values in simulation data (7th column)")
        if args.output_fname is None:
            raise  ValueError("output_fname is not set. Please set it. Note that catalog data will not be created when using original values.")

        sfr = halo_data[:,7]

        print("# Use galaxies with value > {}".format(args.threshold))
        mask = sfr > args.threshold
        pos = pos[mask]
        vel = vel[mask]
        sfr = sfr[mask]

        if args.redshift_space:
            pos_real = copy.deepcopy(pos)
            pos[:,2] += vel[:,2] / rsd_factor(redshift)
            pos_list = [pos_real, pos]
        else:
            pos_list = [pos]
        
        intensities = []
        for pos in pos_list:
            intensity = make_intensity_map(pos, sfr)
            intensities.append(intensity)

        keys = ["intensity", "intensity_rsd"]
        save_hdf5_data(
            data_list = intensities, 
            key_list = keys, 
            fname = args.output_fname,
            args = args,
            )

def add_spherical_offset_and_rsd(
    generated, 
    pos_central,
    vel_central,
    flag_central,
    redshift,
    output_features = [],
    redshift_space = False
):
    distance = get_feature_values(generated, output_features, "SubhaloDist")
    vr = get_feature_values(generated, output_features, "SubhaloVrad")
    vt = get_feature_values(generated, output_features, "SubhaloVtan")

    offset, vz = spherical_offsets_and_vz(
        distance, 
        vr = vr,
        vt = vt,
        flag_central=flag_central
    )

    pos_galaxies = pos_central + offset
        
    if redshift_space:
        pos_galaxies_real = copy.deepcopy(pos_galaxies) 
        pos_galaxies[:,2] += ( vel_central[:,2] + vz ) * rsd_factor(redshift)
        pos_list = [pos_galaxies_real, pos_galaxies]
    else:
        pos_list = [pos_galaxies]

    return pos_list, vz

def get_z_m_p_v_s(input_fname):
    print(f"# Load {input_fname}")

    if input_fname.endswith(".hdf5") or input_fname.endswith(".h5"):
        with h5py.File(args.input_fname, "r") as f:
            redshift = f["Header"].attrs["Redshift"]
            mass = f["Group/GroupMass"][:] #[1e10 Msun/h]
            pos = f["Group/GroupPos"][:] # [kpc/h]
            vel = f["Group/GroupVel"][:] if "Group/GroupVel" in f else None # [km/s]
        
        if vel is None:
            res = np.concatenate([mass[:,None],pos], axis=1)
        else:
            res = np.concatenate([mass[:,None],pos,vel], axis=1)

                
    elif "pinocchio" in input_fname:
        match = re.search(r'pinocchio\.([0-9]+\.[0-9]+)', input_fname)
        redshift = float(match.group(1))
            
        import helpers.ReadPinocchio5 as rp
        mycat = rp.catalog(input_fname)
        
        mass = mycat.data["Mass"] / 1e10 # [1e10 Msun/h]
        pos = mycat.data["pos"]
        vel = mycat.data["vel"]
        
        res = np.concatenate([mass[:,None],pos,vel], axis=1)
    
    else:
        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 

        with open(input_fname, "r") as f:
            first_line = f.readline().strip()
            tokens = first_line.split()
            redshift = float(tokens[1])
        res = np.loadtxt(input_fname)
        # res: logm, x, y, z, vx, vy, vz, sfr

        res[:,0] = 10 ** res[:,1] / 1e10 * hlittle # mass
        res[:,7] = 10 ** res[:,7] # sfr

    return redshift, res

def make_intensity_map(pos, flux, npix, dx_pix):
    if isinstance(npix, int):
        npix = (npix, npix, npix)

    ix_galaxies = (pos / dx_pix).astype(int) # (num_galaxies_valid, 3)    
    valid_mask = np.all((ix_galaxies >= 0) & (ix_galaxies < npix), axis=1)
    ix_valid = ix_galaxies[valid_mask]
    flux_valid = flux[valid_mask]

    intensity = np.zeros(npix)
    np.add.at(intensity, (ix_valid[:, 0], ix_valid[:, 1], ix_valid[:, 2]), flux_valid)

    return intensity


def rsd_factor(redshift):
    H = cosmo.H(redshift).to(u.km/u.s/u.kpc).value #[km/s/kpc]
    hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
    scale_factor = 1 / (1 + redshift)

    return 1. / scale_factor / H * hlittle

if __name__ == "__main__":
    args = parse_args()
    create_data(args)