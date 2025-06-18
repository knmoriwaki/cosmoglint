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

from cosmoglint.utils.generation_utils import save_intensity_data, save_catalog_data

cspeed = 3e10 # [cm/s]
micron = 1e-4 # [cm]

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    ### I/O parameters
    parser.add_argument("--input_fname", type=str, default="./Pinocchio/output/pinocchio.r01000.plc.out", help="Input filename")
    parser.add_argument("--output_fname", type=str, default="test.h5", help="Output filename")

    parser.add_argument("--boxsize", type=float, default=100.0, help="Box size [Mpc/h]")

    ### Output format parameters
    parser.add_argument("--npix", type=int, default=100, help="Number of pixels in x and y direction")
    parser.add_argument("--npix_z", type=int, default=90, help="Number of pixels in z direction")

    parser.add_argument("--redshift_space", action="store_true", default=False, help="Use redshift space")
    parser.add_argument("--gen_both", action="store_true", default=False, help="Generate both real and redshift space data")

    parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass [Msun] to be used")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Galaxies with SFR > threshold [Msun/yr] will be used")

    parser.add_argument("--gen_catalog", action="store_true", default=False, help="Generate a catalog of galaxies instead of a data cube")
    parser.add_argument("--catalog_threshold", type=float, default=10, help="Threshold for SFR in the catalog in [Msun/yr]")

    parser.add_argument("--mass_correction_factor", type=float, default=1.0, help="Mass correction factor; the halo mass is multiplied by this factor before generating galaxies.")

    ### Generative model parameters
    parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 7th column as intensity.")
    parser.add_argument("--prob_threshold", type=float, default=1e-5, help="Below this probability, the galaxy is not generated.")
    parser.add_argument("--max_sfr_file", type=str, default=None, help="File containing maximum IDs for SFR.")

    return parser.parse_args()

def create_data(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    npix = np.array([args.npix, args.npix, args.npix_z])
    dx_pix = args.boxsize / npix

    ### Load data
    print(f"# Load {args.input_fname}")

    if args.gen_both:
        args.redshift_space = True

    if "pinocchio" in args.input_fname:
        match = re.search(r'pinocchio\.([0-9]+\.[0-9]+)', args.input_fname)
        redshift = float(match.group(1))
            
        import cosmoglint.utils.ReadPinocchio5 as rp
        mycat = rp.catalog(args.input_fname)
        
        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 

        logm = np.log10( mycat.data["Mass"] / hlittle ) # [Msun]
        pos = mycat.data["pos"]
        vel = mycat.data["vel"]

    else:
        with open(args.input_fname, "r") as f:
            first_line = f.readline().strip()
            tokens = first_line.split()
            redshift = float(tokens[1])
        data = np.loadtxt(args.input_fname)
        # Input data: logm, x, y, z, vx, vy, vz, value

        logm = data[:, 0]
        pos = data[:, 1:4]
        vel = data[:, 4:7]

    logm += np.log10(args.mass_correction_factor) 

    ### Mask out small halos
    print("# Minimum log mass in catalog [Msun]: {:.5f}".format(np.min(logm)))
    print("# Maximum log mass in catalog [Msun]: {:.5f}".format(np.max(logm)))
    print("# Use halos with log mass [Msun] > {}".format(args.logm_min))
    mask = logm > args.logm_min
    logm = logm[mask]
    pos = pos[mask]
    vel = vel[mask]

    print(f"# Redshift: {redshift}")
    H = cosmo.H(redshift).to(u.km/u.s/u.Mpc).value #[km/s/Mpc]
    hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
    scale_factor = 1 / (1 + redshift)

    if args.model_dir == None:
        print("# Use original values in simulation data (7th column)")
        print("# Use galaxies with value > {}".format(args.threshold))
        value = 10 ** data[:,7]
        value = value[mask]

        if args.gen_both:
            pos_real = copy.deepcopy(pos)

        if args.redshift_space:
            pos[:,2] += vel[:,2] / scale_factor / H * hlittle

        if args.gen_both:
            pos_list = [pos_real, pos]
        else:
            pos_list = [pos]

        if args.gen_catalog:
            pos_valid = []
            for p in pos_list:
                valid_mask = value > args.catalog_threshold
                pos_valid = p[valid_mask]
                value_valid = value[valid_mask]

            save_catalog_data(pos_valid, value_valid, args, args.output_fname)

        else:        
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

            save_intensity_data(intensities, args, args.output_fname)

    else:
        if "Transformer_NF" in args.model_dir:
            from cosmoglint.utils.generation_utils import generate_galaxy_TransNF
            generated, pos_central, vel_central, flag_central = generate_galaxy_TransNF(args, logm, pos, vel)
        else:
            from cosmoglint.utils.generation_utils import generate_galaxy
            generated, pos_central, vel_central, flag_central = generate_galaxy(args, logm, pos, vel)

        sfr = generated[:,0]
        distance = generated[:,1]
    
        num_gal = len(sfr)

        ### Determine positions of galaxies
        print("# Generate positions of galaxies")
        phi = np.random.uniform(0, 2 * np.pi, size=num_gal)
        cos_theta = np.random.uniform(-1, 1, size=num_gal)
        sin_theta = np.sqrt(1 - cos_theta ** 2)    

        pos_galaxies = pos_central
        pos_galaxies[:,0] += distance * sin_theta * np.cos(phi)
        pos_galaxies[:,1] += distance * sin_theta * np.sin(phi)
        pos_galaxies[:,2] += distance * cos_theta

        if args.gen_both: # copy the position in real space before adding redshift space distortion
            pos_galaxies_real = copy.deepcopy(pos_galaxies) 

        if args.redshift_space:
            relative_vel_rad = generated[:,2]
            relative_vel_tan = generated[:,3]
            relative_vel_rad[flag_central] = 0 # Set vr to 0 for central galaxies
            alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
            vz_gal = - relative_vel_rad * cos_theta + relative_vel_tan * sin_theta * np.cos(alpha)
            pos_galaxies[:,2] += ( vel_central[:,2] + vz_gal )/ scale_factor / H * hlittle

        if args.gen_both:
            pos_list = [pos_galaxies_real, pos_galaxies]
        else:
            pos_list = [pos_galaxies]

        def make_intensity_map(pos, flux):
            ix_galaxies = (pos / dx_pix).astype(int) # (num_galaxies_valid, 3)    
            valid_mask = np.all((ix_galaxies >= 0) & (ix_galaxies < npix), axis=1)
            ix_valid = ix_galaxies[valid_mask]
            flux_valid = flux[valid_mask]

            intensity = np.zeros((args.npix, args.npix, args.npix_z))
            np.add.at(intensity, (ix_valid[:, 0], ix_valid[:, 1], ix_valid[:, 2]), flux_valid)

            return intensity

        if args.gen_catalog:
            print("# Generate catalog of galaxies")
            pos_valid = []
            for pos in pos_list:
                valid_mask = sfr > args.catalog_threshold
                pos_valid.append(pos[valid_mask])
                sfr_valid = sfr[valid_mask]
            save_catalog_data(pos_valid, sfr_valid, args, args.output_fname)

        else:
            print("# Assign galaxies to pixels")

            intensities = []
            for pos in pos_list:
                intensity = make_intensity_map(pos, sfr)
                intensities.append(intensity)

            save_intensity_data(intensities, args, args.output_fname)


if __name__ == "__main__":
    args = parse_args()
    create_data(args)