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

from cosmoglint.utils.io_utils import save_hdf5_intensity_data, load_global_params
from cosmoglint.utils import normalize, namespace_to_dict

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


def my_save_catalog_data(pos_list, value, args, output_fname):
    if not isinstance(pos_list, list):
        pos_list = [pos_list]

    with open(output_fname, 'w') as f:
        for i, v in enumerate(value):
            f.write(f"{pos_list[0][i, 0]} {pos_list[0][i, 1]} ")
            for pos in pos_list:
                f.write(f"{pos[i, 2]} ")

            f.write(f"{v}\n")

    print(f"# Catalog saved to {output_fname}")

def create_data(args):
    if (args.output_fname is None) and (args.output_catalog_fname is None):
        raise ValueError("Please set at least one of the output_fname or output_catalog_fname.")
    
    import astropy.units as u

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
    print(f"# Load {args.input_fname}")

    hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 

    if args.input_fname.endswith(".hdf5") or args.input_fname.endswith(".h5"):
        with h5py.File(args.input_fname, "r") as f:
            redshift = f["Header"].attrs["Redshift"]
            mass = f["Group/GroupMass"][:] # [1e10 Msun/h]
            pos = f["Group/GroupPos"][:] # [kpc/h]
            if args.redshift_space:
                vel = f["Group/GroupVel"][:] # [km/s]

    elif "pinocchio" in args.input_fname:
        match = re.search(r'pinocchio\.([0-9]+\.[0-9]+)', args.input_fname)
        redshift = float(match.group(1))
            
        import ReadPinocchio5 as rp
        mycat = rp.catalog(args.input_fname)
        
        mass = mycat.data["Mass"] / 1e10 # [1e10 Msun/h]
        pos = mycat.data["pos"]
        if args.redshift_space:
            vel = mycat.data["vel"]
    
    else:
        with open(args.input_fname, "r") as f:
            first_line = f.readline().strip()
            tokens = first_line.split()
            redshift = float(tokens[1])
        data = np.loadtxt(args.input_fname)
        # Input data: logm, x, y, z, vx, vy, vz, value

        mass = 10 ** data[:, 0] / 1e10 * hlittle
        pos = data[:, 1:4]
        if args.redshift_space:
            vel = data[:, 4:7]

    mass *= args.mass_correction_factor

    ### Mask out small halos
    print("# Minimum log mass in catalog [Msun]: {:.5f}".format(np.min(np.log10(mass))))
    print("# Maximum log mass in catalog [Msun]: {:.5f}".format(np.max(np.log10(mass))))
    print("# Use halos with log mass [Msun] > {}".format(args.logm_min))
    mask = (np.log10(mass) + 10 > args.logm_min)
    
    if args.boxsize_to_use < args.boxsize:
        print("# Use a volume at the last corner -- new boxsize: {:.3f}".format(args.boxsize_to_use))
    xmin = args.boxsize - args.boxsize_to_use
    pos = pos - xmin
    mask = mask & (pos > 0).all(axis=-1)
    
    mass = mass[mask]
    cond = mass[:, None]
    pos = pos[mask]
    if args.redshift_space:
        vel = vel[mask]

    print(f"# Redshift: {redshift}")
    import astropy.units as u
    H = cosmo.H(redshift).to(u.km/u.s/u.kpc).value #[km/s/kpc]
    hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
    scale_factor = 1 / (1 + redshift)

    if args.model_dir == None:
        print("# Use original values in simulation data (7th column)")
        print("# Use galaxies with value > {}".format(args.threshold))
        value = 10 ** data[:,7]
        value = value[mask]

        if args.redshift_space:
            pos_real = copy.deepcopy(pos)
            pos[:,2] += vel[:,2] / scale_factor / H * hlittle
            pos_list = [pos_real, pos]
        else:
            pos_list = [pos]

        if args.output_fname is not None:      
            intensities = []
            for p in pos_list:
                intensity = np.zeros((args.npix, args.npix, args.npix_z))

                for i in range(len(p)):
                    if value[i] < args.threshold:
                        continue

                    ix = np.array([p[i,0], p[i,1], p[i,2]]) / dx_pix        
                    if any(ix < 0) or any(ix >= npix):
                        continue
                    
                    intensity[int(ix[0]), int(ix[1]), int(ix[2])] += value[i]

                intensities.append(intensity)

            keys = ["intensity", "intensity_rsd"]
            save_hdf5_intensity_data(intensities, args, keys, args.output_fname)

        if args.output_catalog_fname is not None:
            pos_valid = []
            for p in pos_list:
                valid_mask = value > args.catalog_threshold
                pos_valid = p[valid_mask]
                value_valid = value[valid_mask]

            my_save_catalog_data(pos_valid, value_valid, args, ["SubhaloSFR"], args.output_catalog_fname)


    else:
        device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        ### Load global parameters
        with open("{}/args.json".format(args.model_dir), "r") as f:
            opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
            
        if args.global_param_file is not None:
            global_params = load_global_params(args.global_param_file, opt.global_features)[args.global_param_id] 
        else:
            global_params = None

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

        seq_length = mask.shape[1]
        num_features = generated.shape[-1]
        num_gal = mask.sum()

        ### Define flag_central
        flag_central = np.zeros_like(mask, dtype=bool) # (num_halos, seq_length)
        flag_central[:, 0] = True

        ### Flatten the arrays
        mask = mask.reshape(-1) # (num_halos * seq_length, )
        generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)
        pos_central = np.repeat(pos[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
        vel_central = np.repeat(vel[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
        flag_central = flag_central.reshape(-1) # (num_halos * seq_length, )

        ### Apply mask to arrays
        generated = generated[mask] # (num_galaxies_valid, num_features)
        pos_central = pos_central[mask] # (num_galaxies_valid, 3)
        vel_central = vel_central[mask] # (num_galaxies_valid, 3)
        flag_central = flag_central[mask] # (num_galaxies_valid, )

        ### Distribute galaxies in cube
        print("# Generate positions of galaxies")

        sfr = generated[:,0]
        distance = generated[:,1]

        phi = np.random.uniform(0, 2 * np.pi, size=num_gal)
        cos_theta = np.random.uniform(-1, 1, size=num_gal)
        sin_theta = np.sqrt(1 - cos_theta ** 2)    

        pos_galaxies = pos_central
        pos_galaxies[:,0] += distance * sin_theta * np.cos(phi)
        pos_galaxies[:,1] += distance * sin_theta * np.sin(phi)
        pos_galaxies[:,2] += distance * cos_theta

        pos_galaxies_real = copy.deepcopy(pos_galaxies) 

        ### Add redshift-space distortion
        if args.redshift_space:
            H = cosmo.H(redshift).to(u.km/u.s/u.kpc).value #[km/s/kpc]
            hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
            scale_factor = 1 / (1 + redshift)

            relative_vel_rad = generated[:,2]
            relative_vel_tan = generated[:,3]
            relative_vel_rad[flag_central] = 0 # Set vr to 0 for central galaxies
            alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
            vz_gal = - relative_vel_rad * cos_theta + relative_vel_tan * sin_theta * np.cos(alpha)
            pos_galaxies[:,2] += ( vel_central[:,2] + vz_gal )/ scale_factor / H * hlittle
        
            pos_list = [pos_galaxies_real, pos_galaxies]
        else:
            pos_list = [pos_galaxies]

        ### Save

        if args.output_fname is not None:
            print("# Assign galaxies to pixels")
            def make_intensity_map(pos, flux):
                ix_galaxies = (pos / dx_pix).astype(int) # (num_galaxies_valid, 3)    
                valid_mask = np.all((ix_galaxies >= 0) & (ix_galaxies < npix), axis=1)
                ix_valid = ix_galaxies[valid_mask]
                flux_valid = flux[valid_mask]

                intensity = np.zeros((args.npix, args.npix, args.npix_z))
                np.add.at(intensity, (ix_valid[:, 0], ix_valid[:, 1], ix_valid[:, 2]), flux_valid)

                return intensity
            
            intensities = []
            for pos in pos_list:
                intensity = make_intensity_map(pos, sfr)
                intensities.append(intensity)

            keys = ["intensity", "intensity_rsd"]
            save_hdf5_intensity_data(intensities, args, keys, args.output_fname)
        
        if args.output_catalog_fname is not None:
            print("# Generate catalog of galaxies")
            pos_valid = []
            for pos in pos_list:
                valid_mask = sfr > args.catalog_threshold
                pos_valid.append(pos[valid_mask])
                sfr_valid = sfr[valid_mask]
            my_save_catalog_data(pos_valid, sfr_valid, args, opt.output_features, args.output_catalog_fname)



if __name__ == "__main__":
    args = parse_args()
    create_data(args)