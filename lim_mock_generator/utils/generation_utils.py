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

def generate_galaxy_TransNF(args, logm, pos, vel):
    """
    args: args.gpu_id, args.model_dir, and args.threshold are used
    logm: (num_halos, ), log mass of the halos
    pos: (num_halos, 3), positions of the halo centers
    vel: (num_halos, 3), velocities of the halo centers
    """

    print("# Use Transformer-NF to generate galaxies")

    from lim_mock_generator.model.transformer_nf import my_model, my_flow_model, generate
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    norm_params = np.array(opt.norm_params)
    xmin = norm_params[:,0]
    xmax = norm_params[:,1] 

    model = my_model(opt)
    model.to(device)
    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir)))
    model.eval()
    print(model)

    flow = my_flow_model(opt)
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
        generated_batch = generate(model, flow, logm_batch, stop_criterion=stop_criterion) # (batch_size, max_length, num_features)
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

def generate_galaxy_two_step(args, logm, pos, vel):
    """
    args: args.gpu_id, args.model_dir, and args.threshold are used
    logm: (num_halos, ), log mass of the halos
    pos: (num_halos, 3), positions of the halo centers
    vel: (num_halos, 3), velocities of the halo centers
    """


    print("# Use Transformer to generate SFR")

    from lim_mock_generator.model.transformer import my_model
    from lim_mock_generator.model.nn import my_NN_model 
    from lim_mock_generator.model.nf import my_flow_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    norm_params = np.array(opt.norm_params)
    xmin = norm_params[:,0]
    xmax = norm_params[:,1]

    model = my_model(opt)
    model.to(device)

    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir)))
    model.eval()
    print(model)

    ### generate SFR
    print("# Generate SFR (batch size: {:d})".format(opt.batch_size))
    logm = (logm - xmin[0]) / (xmax[0] - xmin[0])
    logm = torch.from_numpy(logm).float().to(device)
    
    num_batch = (len(logm) + opt.batch_size - 1) // opt.batch_size
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        logm_batch = logm[start: start + opt.batch_size] # (batch_size, 1)
        with torch.no_grad():
            generated_batch, _ = model.generate(logm_batch)
        generated.append(generated_batch)
    generated = torch.cat(generated, dim=0) # (num_halos, seq_length, 1)

    ### Select valid galaxies
    print("# Select valid galaxies")

    batch, seq_length, num_features_out = generated.shape

    # Set mask for selection
    sfr = generated[:,:,0].cpu().detach().numpy()
    sfr = sfr * (xmax[1] - xmin[1]) + xmin[1]
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
    sfr = sfr.reshape(-1) # (num_halos * seq_length, )

    # Flatten torch tensor
    logm_seq = logm[:,None,None].expand(-1, seq_length, 1).contiguous().view(-1, 1) # (num_halos * seq_length, 1)
    generated = generated.view(-1, num_features_out) # (num_halos * seq_length, 1)
    x_NN = torch.cat([logm_seq, generated], dim=-1) # (num_halos * seq_length, 2)

    # Apply mask to arrays
    flag_central = flag_central[mask] # (num_galaxies_valid, )
    pos_central = pos_central[mask] # (num_galaxies_valid, 3)
    vel_central = vel_central[mask] # (num_galaxies_valid, 3)
    sfr = sfr[mask] # (num_galaxies_valid, )
    
    # Apply mask to tensor
    mask = torch.from_numpy(mask).to(device)
    x_NN = x_NN[mask] # (num_galaxies_valid, 2)

    num_gal = len(x_NN)
    print("# Number of valid galaxies: {:d}".format(num_gal))
    
    ### Load NN
    with open("{}/args.json".format(args.NN_model_dir), "r") as f:
        opt_NN = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt_NN: ", opt_NN)

    if "NN" in args.NN_model_dir:
        model_NN = my_NN_model(opt_NN)
    else:
        model_NN = my_flow_model(opt_NN)

    model_NN.to(device)
    model_NN.load_state_dict(torch.load("{}/model.pth".format(args.NN_model_dir)))
    model_NN.eval()
    print(model_NN)

    ### Generate other properties
    print("# Generate other properties (batch size: {:d})".format(opt_NN.batch_size))
    num_batch = (num_gal + opt_NN.batch_size - 1) // opt_NN.batch_size
    generated_NN = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt_NN.batch_size 
        x_NN_batch = x_NN[start: start + opt_NN.batch_size] # (batch_size, num_features_out)
        with torch.no_grad():
            if "NN" in args.NN_model_dir:
                generated_NN_batch, _ = model_NN.generate(x_NN_batch) # (batch_size, num_features_out)
            else:
                generated_NN_batch = model_NN.sample(1, x_NN_batch) # (batch_size, 1, num_features_out)
                generated_NN_batch = generated_NN_batch.squeeze(1) # (batch_size, num_features_out)

        generated_NN.append(generated_NN_batch)

    generated_NN = torch.cat(generated_NN, dim=0) # (num_galaxies_valid, num_features_out)
    generated_NN = generated_NN.cpu().detach().numpy()
    generated_NN = generated_NN * (xmax[2:2+opt_NN.num_features_out] - xmin[2:2+opt_NN.num_features_out]) + xmin[2:2+opt_NN.num_features_out]

    for i in range(opt_NN.num_features_out):
        if i == 1:
            generated_NN[:,i] = np.sign( generated_NN[:,i] ) * (10 ** np.abs( generated_NN[:,i] ) - 1 )
        else:
            generated_NN[:,i] = 10 ** generated_NN[:,i]

    generated = np.concatenate([sfr[:,None], generated_NN], axis=1) # (num_galaxies_valid, num_features_out)

    return generated, pos_central, vel_central, flag_central


def generate_galaxy(args, logm, pos, vel):
    """
    args: args.gpu_id, args.model_dir, and args.threshold are used
    logm: (num_halos, ), log mass of the halos
    pos: (num_halos, 3), positions of the halo centers
    vel: (num_halos, 3), velocities of the halo centers
    """

    print("# Use Transformer to generate SFR")

    from lim_mock_generator.model.transformer import my_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    ### load Transformer
    with open("{}/args.json".format(args.model_dir), "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    norm_params = np.array(opt.norm_params)
    xmin = norm_params[:,0]
    xmax = norm_params[:,1]

    model = my_model(opt)
    model.to(device)

    model.load_state_dict(torch.load("{}/model.pth".format(args.model_dir)))
    model.eval()
    print(model)

    ### generate galaxies
    print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))
    logm = (logm - xmin[0]) / (xmax[0] - xmin[0])
    logm = torch.from_numpy(logm).float().to(device)
    
    num_batch = (len(logm) + opt.batch_size - 1) // opt.batch_size
    prob_threshold = 1e-5
    stop_criterion = ( np.log10(args.threshold) - xmin[1] ) / (xmax[1] - xmin[1]) # stop criterion for SFR
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 
        logm_batch = logm[start: start + opt.batch_size] # (batch_size, num_features)
        with torch.no_grad():
            generated_batch, _ = model.generate(logm_batch, prob_threshold=prob_threshold, stop_criterion=stop_criterion, cutoff=True) # (batch_size, seq_length, num_features)
        generated.append(generated_batch)
    generated = torch.cat(generated, dim=0) # (num_halos, seq_length, num_features)

    ### Select valid galaxies
    print("# Select valid galaxies")

    # Set mask for selection
    batch, seq_length, num_features = generated.shape

    generated = generated.cpu().detach().numpy()
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


def load_old_plc(filename):
    import struct

    plc_struct_format = "<Q d ddd ddd ddddd"  # Q=uint64, d=double, little-endian
    plc_size = struct.calcsize(plc_struct_format)

    M_list = []
    th_list = []
    ph_list = []
    vl_list = []
    zo_list = []
    with open(filename, "rb") as f:
        while True:
            dummy_bytes = f.read(4)
            if not dummy_bytes:
                break  # EOF
            dummy = struct.unpack("<i", dummy_bytes)[0]

            plc_bytes = f.read(dummy)
            if len(plc_bytes) != dummy:
                break  # 不完全な読み込み

            data = struct.unpack(plc_struct_format, plc_bytes)
            (
                id, z, x1, x2, x3, v1, v2, v3,
                M, th, ph, vl, zo
            ) = data

            dummy2_bytes = f.read(4)
            dummy2 = struct.unpack("<i", dummy2_bytes)[0]

            M_list.append(M)
            th_list.append(th)
            ph_list.append(ph)
            vl_list.append(vl)
            zo_list.append(zo)
    
    return np.array(M_list), np.array(th_list), np.array(ph_list), np.array(vl_list), np.array(zo_list)
