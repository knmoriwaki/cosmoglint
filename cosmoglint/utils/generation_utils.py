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

cspeed = 3e10  # [cm/s]

from .io_utils import normalize, namespace_to_dict

def create_mask(array, threshold): 
    """
    mask out galaxies satisfying either of the following:
    - if it is central and the value is below threshold
    - if it is satellite and any of the satellites before it is below threshold
    """

    _, seq_length = array.shape

    mask_valid = array > threshold # (num_halos, max_length)
    mask_below = array <= threshold # (num_halos, max_length)
    mask_below[:, 0] = False  

    first_below = np.where(mask_below.any(axis=1), mask_below.argmax(axis=1), seq_length)
    indices = np.arange(seq_length)[None, :]  # (1, max_length)
    mask = indices < first_below[:, None]  # (num_halos, max_length)

    mask = mask & mask_valid # (num_halos, max_length)

    return mask

def generate_galaxy(args, x_in, global_params=None, verbose=True):
    """
    args: args.gpu_id, args.model_dir, args.threshold, and args.max_sfr_file are used
    x_in: (num_halos, num_features_in); halo properties
    """

    print("# Use Transformer to generate SFR")

    from cosmoglint.model.transformer import transformer_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        opt.norm_param_dict = namespace_to_dict(opt.norm_param_dict)

    model = transformer_model(opt)
    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir), map_location="cpu"))
    model.to(device)
    model.eval()
    
    if verbose:
        print("opt: ", opt)
        print(model)

    ### generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    for i, key in enumerate(opt.input_features):
        x_in[...,i] = normalize(x_in[...,i], key, opt.norm_param_dict)
    x_in = torch.from_numpy(x_in).float().to(device)

    if global_params is not None:
        global_params = global_params[opt.global_features].to_numpy(dtype=np.float32)
        for i, key in enumerate(opt.global_features):
            global_params[...,i] = normalize(global_params[...,i], key, opt.norm_param_dict)
        global_params = torch.tensor(np.array(global_params), dtype=torch.float32).to(device)

    if args.max_sfr_file is None:
        print("# No max SFR file provided, using default max IDs")
        max_ids = None
    else:
        max_ids = np.loadtxt(args.max_sfr_file)
        max_ids = ( max_ids * opt.num_features_out ).astype(int)
        max_ids = torch.tensor(max_ids).to(device) # (num_features, )
    
    num_batch = (len(x_in) + opt.batch_size - 1) // opt.batch_size
    stop_criterion = normalize(args.threshold, opt.output_features[0], opt.norm_param_dict) # stop criterion for SFR
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        x_batch = x_in[start: start + opt.batch_size] # (batch_size, num_features)
        global_cond_batch = global_params.unsqueeze(0).repeat(len(x_batch), 1) if global_params is not None else None # (batch_size, num_global_features)
        with torch.no_grad():
            generated_batch, _ = model.generate(x_batch, global_cond=global_cond_batch, prob_threshold=1e-5, stop_criterion=stop_criterion, max_ids=max_ids) # (batch_size, seq_length, num_features)
            
        generated.append(generated_batch.cpu().detach().numpy())
        
    generated = np.concatenate(generated, axis=0) # (num_halos, seq_length, num_features) or (num_halos, seq_length * num_features, 1)

    if opt.use_flat_representation:
        generated = generated.squeeze(-1).reshape(len(generated), -1, opt.num_features_in) # (num_halos, max_length, num_features) 
        mask = mask.reshape(len(mask), -1, opt.num_features_in)

    mask = create_mask(generated[:,:,0], stop_criterion) # (num_halos, seq_length)

    # De-normalize
    for i, key in enumerate(opt.output_features):
        generated[...,i] = normalize(generated[...,i], key, opt.norm_param_dict, inverse=True)

    print("# Number of valid galaxies: {:d}".format(len(generated)))
    
    return generated, mask

def generate_galaxy_TransNF(args, x_in, global_params=None, verbose=True):
    """
    args: args.gpu_id, args.model_dir, and args.threshold are used
    x_in: (num_halos, num_features_in), halo properties
    """

    print("# Use Transformer-NF to generate galaxies")

    from cosmoglint.model.transformer_nf import transformer_nf_model, generate_with_transformer_nf
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        opt.norm_param_dict = namespace_to_dict(opt.norm_param_dict)

    model, flow = transformer_nf_model(opt)

    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir), map_location="cpu"))
    model.to(device)
    model.eval()
    
    flow.load_state_dict(torch.load("{}/flow.pth".format(args.model_dir), map_location="cpu"))
    flow.to(device)
    flow.eval()

    if verbose:
        print("opt: ", opt)
        print(model)
        print(flow)

    ### generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    
    for i, key in enumerate(opt.input_features):
        x_in[...,i] = normalize(x_in[...,i], key, opt.norm_param_dict)
    x_in = torch.from_numpy(x_in).float().to(device)

    if global_params is not None:
        global_params = np.array([global_params[name] for name in opt.global_features], dtype=np.float32)
        for i, key in enumerate(opt.global_features):
            global_params[...,i] = normalize(global_params[...,i], key, opt.norm_param_dict)
        global_params = torch.from_numpy(global_params).float().to(device)
    
    num_batch = (len(x_in) + opt.batch_size - 1) // opt.batch_size
    generated = []
    def stop_criterion(sample):
        # sample: (batch, num_features)
        return (sample[:, 0] < 1).all()
    
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        x_batch = x_in[start: start + opt.batch_size] # (batch_size, 1)
        global_cond_batch = global_params.unsqueeze(0).repeat(len(x_batch), 1) if global_params is not None else None
        generated_batch = generate_with_transformer_nf(model, flow, x_batch, global_cond=global_cond_batch, stop_criterion=stop_criterion) # (batch_size, max_length, num_features)
        generated.append(generated_batch.cpu().detach().numpy())
    generated = torch.cat(generated, dim=0) # (num_halos, max_length, num_features) or (num_halos, max_length * num_features, 1)
     
    # De-normalize
    for i, key in enumerate(opt.output_features):
        generated[...,i] = normalize(generated[...,i], key, opt.norm_param_dict, inverse=True)

    # Set mask for selection
    sfr = generated[...,0]
    mask = create_mask(sfr, args.threshold) # (num_halos, seq_length)   
    
    print("# Number of valid galaxies: {:d}".format(len(generated)))
    
    return generated, mask

def populate_galaxies_in_cube(args, x_in, pos, vel, redshift, cosmo, global_params=None):
    
    if "Transformer_NF" in args.model_dir:
        generated, mask = generate_galaxy_TransNF(args, x_in, global_params=global_params)
    else:
        generated, mask = generate_galaxy(args, x_in, global_params=global_params)

    seq_length = mask.shape[1]
    num_features = generated.shape[-1]
    num_gal = mask.sum()

    # Define flag_central
    flag_central = np.zeros_like(mask, dtype=bool) # (num_halos, seq_length)
    flag_central[:, 0] = True

    # Flatten the arrays
    mask = mask.reshape(-1) # (num_halos * seq_length, )
    generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)
    pos_central = np.repeat(pos[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
    vel_central = np.repeat(vel[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
    flag_central = flag_central.reshape(-1) # (num_halos * seq_length, )

    # Apply mask to arrays
    generated = generated[mask] # (num_galaxies_valid, num_features)
    pos_central = pos_central[mask] # (num_galaxies_valid, 3)
    vel_central = vel_central[mask] # (num_galaxies_valid, 3)
    flag_central = flag_central[mask] # (num_galaxies_valid, )

    # Distribute galaxies in cube
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

    # Add redshift-space distortion
    if args.redshift_space:
        import astropy.units as u
        H = cosmo.H(redshift).to(u.km/u.s/u.Mpc).value #[km/s/Mpc]
        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
        scale_factor = 1 / (1 + redshift)

        relative_vel_rad = generated[:,2]
        relative_vel_tan = generated[:,3]
        relative_vel_rad[flag_central] = 0 # Set vr to 0 for central galaxies
        alpha = np.random.uniform(0, 2 * np.pi, size=num_gal)
        vz_gal = - relative_vel_rad * cos_theta + relative_vel_tan * sin_theta * np.cos(alpha)
        pos_galaxies[:,2] += ( vel_central[:,2] + vz_gal )/ scale_factor / H * hlittle

    return sfr, pos_galaxies_real, pos_galaxies

def populate_galaxies_in_lightcone(args, x_in, pos, redshift, cosmo, global_params=None):
    """
    args: args.gpu_id, args.model_dir, args.model_config_file, args.args.threshold, and args.param_dir are used
    x_in: (num_halos, )
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
    
        x_now = x_in[mask_z] # (num_halos_in_bin, )
        pos_now = pos[mask_z] # (num_halos_in_bin, 3)
        redshift_now = redshift[mask_z] # (num_halos_in_bin, 1)

        opt.model_dir = "{}/{}".format(args.model_dir, model_path)
        opt.max_sfr_file = max_sfr_file_list[i]

        if "Transformer_NF" in opt.model_dir:
            generated, mask = generate_galaxy_TransNF(opt, x_now, global_params=global_params, verbose=False)
        else:
            generated, mask = generate_galaxy(opt, x_now, global_params=global_params, verbose=False)
            
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
    from .cosmology_utils import cMpc_to_arcsec, dcMpc_to_dz
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


    return sfr, pos_galaxies, redshift_central_all
