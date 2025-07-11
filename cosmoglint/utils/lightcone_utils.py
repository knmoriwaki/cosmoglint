import os
import sys
import argparse
import json

from tqdm import tqdm
import numpy as np
import torch

import warnings, os

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

def load_lightcone_data(input_fname, cosmo=cosmo_default):
    print(f"# Load {input_fname}")

    if "pinocchio" in input_fname: 
        if "old_version" in input_fname:
            M, theta, phi, _, redshift_obs, redshift_real = load_old_plc(input_fname)
            logm = np.log10(M)
        else:
            import cosmoglint.utils.ReadPinocchio5 as rp
            myplc = rp.plc(input_fname)
            
            logm = np.log10( myplc.data["Mass"] )
            theta = myplc.data["theta"] # [arcsec]
            phi = myplc.data["phi"]

            redshift_obs = myplc.data["obsz"]
            redshift_real = myplc.data["truez"]

        hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
        logm -= np.log10(hlittle) # [Msun]

        theta = ( 90. - theta ) * 3600 # [arcsec]
        pos_x = theta * np.cos( phi * np.pi / 180. ) # [arcsec] 
        pos_y = theta * np.sin( phi * np.pi / 180. ) # [arcsec]

        theta_max = np.max(theta)
        pos_x += theta_max / 1.5
        pos_y += theta_max / 1.5
        
        print("# Minimum log mass in catalog: {:.5f}".format(np.min(logm)))
        print("# Redshift: {:.3f} - {:.3f}".format(np.min(redshift_real), np.max(redshift_real)))
        print("# Number of halos: {}".format(len(logm)))

    else:
        raise ValueError("Unknown input file format")
    
    return logm, pos_x, pos_y, redshift_obs, redshift_real


def load_old_plc(filename):
    import struct

    plc_struct_format = "<Q d ddd ddd ddddd"  # Q=uint64, d=double, little-endian
    plc_size = struct.calcsize(plc_struct_format)

    M_list = []
    th_list = []
    ph_list = []
    vl_list = []
    zo_list = []
    z_list = []
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
            z_list.append(z)
    
    return np.array(M_list), np.array(th_list), np.array(ph_list), np.array(vl_list), np.array(zo_list), np.array(z_list)


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

def generate_galaxy_in_lightcone(args, logm, pos, redshift):
    """
    args: args.gpu_id, args.model_dir, args.model_config_file, args.args.threshold, and args.param_dir are used
    logm: (num_halos, )
    pos: (num_halos, 3)
    redshift: (num_halos, )
    """

    print("# Use Transformer to generate SFR")

    from cosmoglint.model.transformer import transformer_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

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
        mask_z = (bin_indices == i)
        
        print("# Snapshot number: {:d}, Redshift: {:.2f}".format(snapshot_number, redshift_of_snapshot))
        
        ### Skip if no haloes in this redshift bin
        if not np.any(mask_z):
            print("# No haloes in redshift bin {:d} (snapshot number {:d}), skipping...".format(i, snapshot_number))
            continue
    
        ### load Transformer
        model_dir = "{}/{}".format(args.model_dir, model_path)

        with open(f"{model_dir}/args.json", "r") as f:
            opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        #print("opt: ", opt)

        norm_params = np.array(opt.norm_params)
        xmin = norm_params[:,0]
        xmax = norm_params[:,1]

        model = transformer_model(opt)
        model.to(device)

        model.load_state_dict(torch.load("{}/model.pth".format(model_dir)))
        model.eval()
        #print(model)

        ### generate galaxies
        print("# Generate galaxies (batch size: {:d})".format(opt.batch_size))

        max_ids = np.loadtxt(max_sfr_file_list[i])
        max_ids = ( max_ids * opt.num_features_out ).astype(int) 
        max_ids = torch.from_numpy(max_ids).long().to(device)

        logm_now = (logm - xmin[0]) / (xmax[0] - xmin[0])
        logm_now = torch.from_numpy(logm_now).float().to(device)    
        
        logm_now = logm_now[mask_z] # (num_halos_in_bin, )
        pos_now = pos[mask_z] # (num_halos_in_bin, 3)
        redshift_now = redshift[mask_z] # (num_halos_in_bin, )

        prob_threshold = 1e-5
        stop_criterion = ( np.log10(args.threshold) - xmin[1] ) / (xmax[1] - xmin[1]) # stop criterion for SFR

        num_batch = (len(logm_now) + opt.batch_size - 1) // opt.batch_size
        generated = []
        for batch_idx in tqdm(range(num_batch)):
            start = batch_idx * opt.batch_size 
            logm_batch = logm_now[start: start + opt.batch_size] # (batch_size, num_features)
            with torch.no_grad():
                generated_batch, _ = model.generate(logm_batch, prob_threshold=prob_threshold, stop_criterion=stop_criterion, max_ids=max_ids) # (batch_size, seq_length, num_features)
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
        pos_central = np.repeat(pos_now[:,None,:], seq_length, axis=1).reshape(-1, 3) # (num_halos * seq_length, 3)
        redshift_central = np.repeat(redshift_now[:,None], seq_length, axis=1).reshape(-1) # (num_halos * seq_length,)
        generated = generated.reshape(-1, num_features) # (num_halos * seq_length, num_features)

        # Apply mask to arrays
        flag_central = flag_central[mask] # (num_galaxies_valid, )
        pos_central = pos_central[mask] # (num_galaxies_valid, 3)
        redshift_central = redshift_central[mask] # (num_galaxies_valid,)
        generated = generated[mask] # (num_galaxies_valid, num_features)

        for i in range(num_features):
            if i == 2:
                generated[:,i] = np.sign( generated[:,i] ) * (10 ** np.abs( generated[:,i] ) - 1 )
            else:
                generated[:,i] = 10 ** generated[:,i]

        generated_all.append(generated)
        pos_central_all.append(pos_central)
        redshift_central_all.append(redshift_central)
        flag_central_all.append(flag_central)
        
        print("# Number of valid galaxies for snapshot number {:d}: {}".format(snapshot_number, len(generated)))

    generated_all = np.concatenate(generated_all, axis=0) # (num_galaxies_valid, num_features)
    pos_central_all = np.concatenate(pos_central_all, axis=0)
    redshift_central_all = np.concatenate(redshift_central_all, axis=0) # (num_galaxies_valid,)
    flag_central_all = np.concatenate(flag_central_all, axis=0) # (num_galaxies_valid,)
    
    return generated_all, pos_central_all, redshift_central_all, flag_central_all
