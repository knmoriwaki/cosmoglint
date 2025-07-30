import os
import sys
import argparse
import json

from tqdm import tqdm
import numpy as np
import torch

import warnings, os

import copy

def short_formatwarning(msg, category, filename, lineno, line=None):
    return f"{os.path.basename(filename)}:{lineno}: {category.__name__}: {msg}\n"

warnings.formatwarning = short_formatwarning
warnings.filterwarnings("always", category=RuntimeWarning)

import astropy.units as u
from astropy.constants import c as cspeed # [m/s]
from astropy.cosmology import FlatLambdaCDM
cosmo_default = FlatLambdaCDM(H0=67.74, Om0=0.3089)

def cMpc_to_arcsec(l_cMpc, z, cosmo=cosmo_default, l_with_hlittle=False): 
    if l_with_hlittle:
        hlittle = cosmo.H0.value / 100
        l_cMpc = l_cMpc / hlittle # [Mpc/h] -> [Mpc]
    l_rad = l_cMpc * u.Mpc / cosmo.comoving_transverse_distance(z)
    l_arcsec = (l_rad * u.radian).to(u.arcsec)
    return l_arcsec.value

def arcsec_to_cMpc(l_arcsec, z, cosmo=cosmo_default, l_with_hlittle=False):
    l_rad = l_arcsec * u.arcsec / u.radian
    l_cMpc = ( cosmo.comoving_transverse_distance(z) * l_rad ).to(u.Mpc)
    if l_with_hlittle:
        hlittle = cosmo.H0.value / 100
        l_cMpc = l_cMpc * hlittle # [Mpc] -> [Mpc/h]
    return l_cMpc.value 

def dcMpc_to_dz(l_cMpc, z, cosmo=cosmo_default, l_with_hlittle=False):
    if l_with_hlittle:
        hlittle = cosmo.H0.value / 100
        l_cMpc = l_cMpc / hlittle # [Mpc/h] -> [Mpc]
    dx_dz = (cspeed  / cosmo.H(z)).to(u.Mpc)
    d_z = l_cMpc / dx_dz.value 
    return d_z

def dz_to_dcMpc(dz, z, cosmo=cosmo_default, l_with_hlittle=False):
    dx_dz = (cspeed / cosmo.H(z)).to(u.Mpc)
    l_cMpc = dz * dx_dz.value 
    if l_with_hlittle:
        hlittle = cosmo.H0.value / 100
        l_cMpc = l_cMpc * hlittle # [Mpc] -> [Mpc/h]
    return l_cMpc # [Mpc/h] if l_with_hlittle else [Mpc]

def freq_to_comdis(nu_obs, nu_rest, cosmo=cosmo_default, l_with_hlittle=False):
    z = nu_rest / nu_obs - 1
    if z < 0:
        print("Error: z < 0")
        sys.exit(1)
    l_cMpc = cosmo.comoving_distance(z).to(u.Mpc).value
    if l_with_hlittle:
        hlittle = cosmo.H0.value / 100
        l_cMpc = l_cMpc * hlittle # [Mpc] -> [Mpc/h]

    return l_cMpc # [Mpc/h] if l_with_hlittle else [Mpc]

def z_to_log_lumi_dis(z, cosmo=cosmo_default):
    return np.log10( cosmo.luminosity_distance(z).to(u.cm).value )

def populate_galaxies_in_lightcone(args, logm, pos, redshift, cosmo=cosmo_default):
    """
    args: args.gpu_id, args.model_dir, args.model_config_file, args.args.threshold, and args.param_dir are used
    logm: (num_halos, )
    pos: (num_halos, 3)
    redshift: (num_halos, )
    """

    print("# Use Transformer to generate SFR")

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
    bin_indices = np.digitize(redshift, bin_edges)  

    if args.param_dir is None:
        max_sfr_file_list = [ None for snapshot_number in snapshot_dict ]
    else:
        max_sfr_file_list = ["{}/max_nbin20_{:d}.txt".format(args.param_dir, snapshot_number) for snapshot_number in snapshot_dict]

    for i, snapshot_number in enumerate(snapshot_dict):
        model_path, redshift_of_snapshot = snapshot_dict[snapshot_number]
        print("# Snapshot number: {:d}, Redshift: {:.2f}".format(snapshot_number, redshift_of_snapshot))
        
        ### Skip if no haloes in this redshift bin
        mask_z = (bin_indices == i)
        if not np.any(mask_z):
            print("# No haloes in redshift bin {:d} (snapshot number {:d}), skipping...".format(i, snapshot_number))
            continue
    
        logm_now = logm[mask_z] # (num_halos_in_bin, )
        pos_now = pos[mask_z] # (num_halos_in_bin, 3)
        redshift_now = redshift[mask_z] # (num_halos_in_bin, 1)

        opt.model_dir = "{}/{}".format(args.model_dir, model_path)
        opt.max_sfr_file = max_sfr_file_list[i]
        opt.prob_threshold = 1e-5

        if "Transformer_NF" in opt.model_dir:
            raise ValueError("Transformer_NF model is not supported yet. Please use a different model.")
        else:
            from .generation_utils import generate_galaxy
            generated, mask = generate_galaxy(opt, logm_now)
            
        seq_length = mask.shape[1]
        num_features = generated.shape[-1]

        # Define flag_central
        flag_central = np.zeros_like(mask, dtype=bool)
        flag_central[:, 0] = True

        # Flatten the arrays
        mask = mask.reshape(-1)
        generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)
        pos_central = np.repeat(pos_now[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
        redshift_central = np.repeat(redshift_now[:,None], seq_length, axis=1).reshape(-1) # (num_halos * seq_length, 3)
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
    
    ### Distributes galaxies in lightcone
    sfr = generated_all[:,0]
    distance = generated_all[:,1]

    num_gal = len(sfr)

    # Determine positions of galaxies
    print("# Generate positions of galaxies")
    _phi = np.random.uniform(0, 2 * np.pi, size=num_gal)
    _cos_theta = np.random.uniform(-1, 1, size=num_gal)
    _sin_theta = np.sqrt(1 - _cos_theta ** 2)
    
    # Convert Mpc to deg
    distance_arcsec = cMpc_to_arcsec(distance, redshift_central_all, cosmo=cosmo, l_with_hlittle=True)
    distance_z = dcMpc_to_dz(distance, redshift_central_all, cosmo=cosmo, l_with_hlittle=True)

    pos_galaxies = pos_central
    pos_galaxies[:,0] += distance_arcsec * _sin_theta * np.cos(_phi)
    pos_galaxies[:,1] += distance_arcsec * _sin_theta * np.sin(_phi)
    pos_galaxies[:,2] += distance_z * _cos_theta
    
    # Add redshift-space distortion
    if args.redshift_space:

        relative_vel_rad = generated[:,2]
        relative_vel_tan = generated[:,3]
        relative_vel_rad[flag_central] = 0 # Set vr to 0 for central galaxies
        alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
        vz_gal = - relative_vel_rad * _cos_theta + relative_vel_tan * _sin_theta * np.cos(alpha)
        
        beta = vz_gal / (cspeed * 100) # [(km/s) / (km/s)]

        redshift_rest = pos_galaxies[:,2]
        pos_galaxies[:,2] = ( 1. + redshift_rest ) * np.sqrt( (1. + beta) / (1. - beta) ) - 1.0


    return sfr, pos_galaxies, redshift_central_all
