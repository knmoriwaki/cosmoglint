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

from cosmoglint.utils.io_utils import normalize, namespace_to_dict

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

def sample_galaxies(args, x_in, global_params=None, verbose=True):
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

    ### Format input data
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
    
    ### Generate galaxies
    num_batch = (len(x_in) + opt.batch_size - 1) // opt.batch_size
    stop_criterion = normalize(args.threshold, opt.output_features[0], opt.norm_param_dict) # stop criterion for SFR
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        x_batch = x_in[start: start + opt.batch_size] # (batch_size, num_features)
        global_cond_batch = global_params.unsqueeze(0).repeat(len(x_batch), 1) if global_params is not None else None # (batch_size, num_global_features)
        with torch.no_grad():
            generated_batch, _ = model.generate(x_batch, global_cond=global_cond_batch, prob_threshold=1e-5, stop_criterion=stop_criterion, max_ids=max_ids, monotonicity_start_index=args.monotonicity_start_index) # (batch_size, seq_length, num_features)
            
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

def sample_galaxies_TransNF(args, x_in, global_params=None, verbose=True):
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
