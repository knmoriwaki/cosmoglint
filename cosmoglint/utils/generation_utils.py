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

def save_catalog_data(pos_list, value, args, output_fname):
    if not isinstance(pos_list, list):
        pos_list = [pos_list]

    with open(output_fname, 'w') as f:
        for i, v in enumerate(value):
            f.write(f"{pos_list[0][i, 0]} {pos_list[0][i, 1]} ")
            for pos in pos_list:
                f.write(f"{pos[i, 2]} ")

            f.write(f"{v}\n")

    print(f"# Catalog saved to {output_fname}")

def save_intensity_data(intensity, args, output_fname):
    args_dict = vars(args)
    args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}

    if not isinstance(intensity, list):
        intensity = [intensity]
    
    with h5py.File(output_fname, 'w') as f:
        for i, d in enumerate(intensity):
            f.create_dataset(f'intensity{i}', data=d)    
        for key, value in args_dict.items():
            f.attrs[key] = value
    print(f"# Data cube saved as {output_fname}")


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

def generate_galaxy(args, logm, pos, vel):
    """
    args: args.gpu_id, args.model_dir, args.threshold, args.prob_threshold, and args.max_sfr_file are used
    logm: (num_halos, ), log mass of the halos
    pos: (num_halos, 3), positions of the halo centers
    vel: (num_halos, 3), velocities of the halo centers
    """

    print("# Use Transformer to generate SFR")

    from cosmoglint.model.transformer import transformer_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    norm_params = np.array(opt.norm_params)
    xmin = norm_params[:,0]
    xmax = norm_params[:,1]

    model = transformer_model(opt)
    model.to(device)

    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir)))
    model.eval()
    print(model)

    ### generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    logm = (logm - xmin[0]) / (xmax[0] - xmin[0])
    logm = torch.from_numpy(logm).float().to(device)

    if args.max_sfr_file is None:
        max_ids = None
    else:
        max_ids = np.loadtxt(args.max_sfr_file)
        max_ids = ( max_ids * opt.num_features_out ).astype(int)
        max_ids = torch.tensor(max_ids).to(device) # (num_features, )
    
    num_batch = (len(logm) + opt.batch_size - 1) // opt.batch_size
    stop_criterion = ( np.log10(args.threshold) - xmin[1] ) / (xmax[1] - xmin[1]) # stop criterion for SFR
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        logm_batch = logm[start: start + opt.batch_size] # (batch_size, num_features)
        with torch.no_grad():
            generated_batch, _ = model.generate(logm_batch, prob_threshold=args.prob_threshold, stop_criterion=stop_criterion, max_ids=max_ids) # (batch_size, seq_length, num_features)
        generated.append(generated_batch.cpu().detach().numpy())
    generated = np.concatenate(generated, axis=0) # (num_halos, seq_length, num_features)

    ### Select valid galaxies
    print("# Select valid galaxies")

    # Set mask for selection
    batch, seq_length, num_features = generated.shape

    generated = generated * (xmax[1:1+num_features] - xmin[1:1+num_features]) + xmin[1:1+num_features]

    sfr = generated[:,:,0]
    sfr = 10 ** sfr # (num_halos, seq_length)
    mask = create_mask(sfr, args.threshold) # (num_halos, seq_length)

    # Define flag_central
    flag_central = np.zeros_like(sfr, dtype=bool)
    flag_central[:, 0] = True

    # Flatten the arrays
    mask = mask.reshape(-1) # (num_halos * seq_length, ) 
    flag_central = flag_central.reshape(-1)
    pos_central = np.repeat(pos[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
    vel_central = np.repeat(vel[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
    generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)

    # Apply mask to arrays
    flag_central = flag_central[mask] # (num_galaxies_valid, )
    pos_central = pos_central[mask] # (num_galaxies_valid, 3)
    vel_central = vel_central[mask] # (num_galaxies_valid, 3)
    generated = generated[mask] # (num_galaxies_valid, num_features)

    for i in range(num_features):
        if i == 2:
            generated[:,i] = np.sign( generated[:,i] ) * (10 ** np.abs( generated[:,i] ) - 1 )
        else:
            generated[:,i] = 10 ** generated[:,i]
    
    print("# Number of valid galaxies: {:d}".format(len(generated)))
    
    return generated, pos_central, vel_central, flag_central

def generate_galaxy_TransNF(args, logm, pos, vel):
    """
    args: args.gpu_id, args.model_dir, and args.threshold are used
    logm: (num_halos, ), log mass of the halos
    pos: (num_halos, 3), positions of the halo centers
    vel: (num_halos, 3), velocities of the halo centers
    """

    print("# Use Transformer-NF to generate galaxies")

    from cosmoglint.model.transformer_nf import transformer_nf_model, generate_with_transformer_nf
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    norm_params = np.array(opt.norm_params)
    xmin = norm_params[:,0]
    xmax = norm_params[:,1] 

    model, flow = transformer_nf_model(opt)

    model.to(device)
    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir)))
    model.eval()
    print(model)

    flow.to(device)
    flow.load_state_dict(torch.load("{}/flow.pth".format(args.model_dir)))
    flow.eval()
    print(flow)

    ### generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    logm = (logm - xmin[0]) / (xmax[0] - xmin[0])
    logm = torch.from_numpy(logm).float().to(device)
    
    num_batch = (len(logm) + opt.batch_size - 1) // opt.batch_size
    generated = []
    def stop_criterion(sample):
        # sample: (batch, num_features)
        return (sample[:, 0] < 1).all()
    
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        logm_batch = logm[start: start + opt.batch_size] # (batch_size, 1)
        generated_batch = generate_with_transformer_nf(model, flow, logm_batch, stop_criterion=stop_criterion) # (batch_size, max_length, num_features)
        generated.append(generated_batch)
    generated = torch.cat(generated, dim=0) # (num_halos, max_length, num_features)
    generated = generated.cpu().detach().numpy()

    ### Select valid galaxies
    print("# Select valid galaxies")

    batch, seq_length, num_features = generated.shape

    # Set mask for selection -- sfr > threshold
    # This mask is applied for both central and satellite -- this is the minimum sfr that the model can generate
    sfr = generated[:,:,0]
    sfr = sfr * (xmax[1] - xmin[1]) + xmin[1]
    sfr = 10 ** sfr # (num_halos, seq_length)
    mask = create_mask(sfr, args.threshold) # (num_halos, seq_length)

    # Define flag_central
    flag_central = np.zeros_like(sfr, dtype=bool)
    flag_central[:, 0] = True

    # Flatten the arrays
    mask = mask.reshape(-1) # (num_halos * seq_length, ) 
    flag_central = flag_central.reshape(-1)
    pos_central = np.repeat(pos[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * max_length, 3)
    vel_central = np.repeat(vel[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * max_length, 3)
    generated = generated.reshape(-1, num_features) # (num_halos * max_length, num_features)
    
    # Apply mask to arrays
    flag_central = flag_central[mask] # (num_galaxies_valid, )
    pos_central = pos_central[mask] # (num_galaxies_valid, 3)
    vel_central = vel_central[mask] # (num_galaxies_valid, 3)
    generated = generated[mask] # (num_galaxies_valid, num_features)

    num_gal = len(flag_central)
    print("# Number of valid galaxies: {:d}".format(num_gal))
    
    ### Denormalize
    generated = generated * (xmax[1:1+num_features] - xmin[1:1+num_features]) + xmin[1:1+num_features]
    for i in range(num_features):
        if i == 2:
            generated[:,i] = np.sign( generated[:,i] ) * (10 ** np.abs( generated[:,i] ) - 1 )
        else:
            generated[:,i] = 10 ** generated[:,i]

    return generated, pos_central, vel_central, flag_central
