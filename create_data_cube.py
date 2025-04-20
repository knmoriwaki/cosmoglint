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

parser = argparse.ArgumentParser()

parser.add_argument("--input_fname", type=str, default="../Pinocchio/output/pinocchio.r01000.plc.out")
parser.add_argument("--output_fname", type=str, default="test.h5")

parser.add_argument("--boxsize", type=float, default=100.0)
parser.add_argument("--lambda_start", type=float, default=1300.0)
parser.add_argument("--npix", type=int, default=100)
parser.add_argument("--npix_z", type=int, default=90)
parser.add_argument("--seed", type=int, default=12345)

parser.add_argument("--redshift_space", action="store_true", default=False, help="Use redshift space")

parser.add_argument("--logm_min", type=float, default=11.0, help="Minimum log mass")
parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for SFR")

parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 4th column as intensity.")
parser.add_argument("--NN_model_dir", type=str, default=None, help="The directory of the NN model. If not given, use 4th column as intensity.") 

args = parser.parse_args()

### Set up
npix = np.array([args.npix, args.npix, args.npix_z])
dx_pix = args.boxsize / npix

""" # Should use this for realistic data
id_line = ["co_2_1"]
lambda_rest = [1300.9] # [um]
wavelength = np.zeros(args.npix_z)
frequency = np.zeros(args.npix_z)
wavelength[0] = args.lambda_start
frequency[0] = cspeed / (wavelength[0] * micron)
for iz in range(1, args.npix_z):
    wavelength[iz] = wavelength[iz-1] * (1. + 1. / args.spec_res)
    frequency[iz] = cspeed / (wavelength[iz] * micron)
zlist = [ [ w / lam_rest - 1. for w in wavelength] for lam_rest in lambda_rest] # (nlines, npix_z)
"""

### Load data
print(f"# Load {args.input_fname}")

if "pinocchio" in args.input_fname:
    match = re.search(r'pinocchio\.([0-9]+\.[0-9]+)', args.input_fname)
    redshift = float(match.group(1))
        
    import Pinocchio.scripts.ReadPinocchio5 as rp
    mycat = rp.catalog(args.input_fname)
    
    logm = np.log10( mycat.data["Mass"] )
    pos = mycat.data["pos"]
    vel = mycat.data["vel"]
    
    print(f"# Minimum log mass in catalog: {np.min(logm)}")

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

### Mask out small halos
mask = logm > args.logm_min
logm = logm[mask]
pos = pos[mask]
vel = vel[mask]

print(f"# Redshift: {redshift}")
H = cosmo.H(redshift).to(u.km/u.s/u.Mpc).value #[km/s/Mpc]
hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
a = 1 / (1 + redshift)

if args.redshift_space:
    print(f"# Map in redshift space")
    pos[:,2] += vel[:,2] / a / H * hlittle # [Mpc/h]

intensity = np.zeros((args.npix, args.npix, args.npix_z))

if args.model_dir == None:
    print("# Use original values in simulation data (7th column)")
    value = 10 ** data[:,7]
    for i in range(len(pos)):
        if value[i] < args.threshold:
            continue

        ix = np.array([pos[i,0], pos[i,1], pos[i,2]]) / dx_pix        
        if any(ix < 0) or any(ix >= npix):
            continue
        
        intensity[int(ix[0]), int(ix[1]), int(ix[2])] += value[i]

else:
    if "Transformer_NF" in args.model_dir:
        from utils import generate_galaxy_TransNF
        generated, pos_central, vel_central, flag_central = generate_galaxy_TransNF(args, logm, pos, vel)

    else:
        from utils import generate_galaxy
        generated, pos_central, vel_central, flag_central = generate_galaxy(args, logm, pos, vel)

    sfr = generated[:,0]
    distance = generated[:,1]
    relative_vel_rad = generated[:,2]
    relative_vel_tan = generated[:,3]

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
    if args.redshift_space:
        relative_vel_rad[flag_central] = 0
        alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
        vz_gal = - relative_vel_rad * cos_theta + relative_vel_tan * sin_theta * np.cos(alpha)
        pos_galaxies[:,2] += vz_gal / a / H * hlittle

    ix_galaxies = (pos_galaxies / dx_pix).astype(int) # (num_galaxies_valid, 3)
    
    ## Assign galaxies
    print("# Assign galaxies")
    valid_mask = np.all((ix_galaxies >= 0) & (ix_galaxies < npix), axis=1)
    ix_valid = ix_galaxies[valid_mask]
    sfr_valid = sfr[valid_mask]

    np.add.at(intensity, (ix_valid[:, 0], ix_valid[:, 1], ix_valid[:, 2]), sfr_valid)

args_dict = vars(args)
args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
    
with h5py.File(args.output_fname, 'w') as f:
    f.create_dataset('intensity', data=intensity)
    
    for key, value in args_dict.items():
        f.attrs[key] = value

print(f"Data cube saved as {args.output_fname}")

    