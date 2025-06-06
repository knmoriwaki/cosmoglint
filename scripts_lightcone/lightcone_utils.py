import os
import sys
import argparse
import json
import copy
import h5py
import re

from tqdm import tqdm

import numpy as np
from numpy.fft import fftn, fftfreq

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import torch

import astropy.units as u
from astropy.constants import c as cspeed # [m/s]
from astropy.cosmology import FlatLambdaCDM
cosmo_default = FlatLambdaCDM(H0=67.74, Om0=0.3089)


micron = 1e-4 # [cm]
GHz = 1e9
Jy = 1.0e-23        # jansky (erg/s/cm2/Hz)
arcsec = 4.848136811094e-6 # [rad] ... arcmin / 60 //


def arcsec_to_cMpc(l_arcsec, z, cosmo=cosmo_default):
    l_rad = l_arcsec * u.arcsec / u.radian
    l_cMpc = ( cosmo.comoving_transverse_distance(z) * l_rad ).to(u.Mpc)
    return l_cMpc.value 

def freq_to_comdis(nu_obs, nu_rest, cosmo=cosmo_default):
    z = nu_rest / nu_obs - 1
    if z < 0:
        print("Error: z < 0")
        sys.exit(1)
    return cosmo.comoving_distance(z).to(u.Mpc).value

def cMpc_to_arcsec(l_cMpc, z, cosmo, l_with_hlittle=True): 
    hlittle = cosmo.H0.value / 100
    if l_with_hlittle:
        l_cMpc = l_cMpc / hlittle # [Mpc/h] -> [Mpc]
    l_rad = l_cMpc * u.Mpc / cosmo.comoving_transverse_distance(z)
    l_arcsec = (l_rad * u.radian).to(u.arcsec)
    return l_arcsec.value

def dcMpc_to_dz(l_cMpc, z, cosmo, l_with_hlittle=True):
    hlittle = cosmo.H0.value / 100
    if l_with_hlittle:
        l_cMpc = l_cMpc / hlittle # [Mpc/h] -> [Mpc]
    dx_dz = (cspeed  / cosmo.H(z)).to(u.Mpc)
    d_z = l_cMpc / dx_dz.value 
    return d_z

def z_to_log_lumi_dis(z, cosmo):
    return np.log10( cosmo.luminosity_distance(z).to(u.cm).value )

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


def generate_galaxy(args, logm, pos, redshift):
    """
    args: args.gpu_id, args.model_dir, args.threshold, and args.param_dir are used
    logm: (num_halos, )
    pos: (num_halos, 3)
    redshift: (num_halos, )
    """

    print("# Use Transformer to generate SFR")

    from lim_mock_generator.model.transformer import my_model
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    generated_all = []
    pos_central_all = [] 
    redshift_central_all = [] 
    flag_central_all = []

    ### Divide haloes into redshift bins
    snapshot_number_list = [67, 
                            49, 
                            33, 
                            25, 
                            21]
    suffix_list = ["_ep40_bs512_w0.02", 
                   "_ep40_bs512_w0.02", 
                   "_ep40_bs512_w0.02", 
                   "_ep60_bs512_w0.02", 
                   "_ep60_bs512_w0.02", ]
    max_ids_file_list = ["{}/max_ids_20_{:d}.txt".format(args.param_dir, snapshot_number) for snapshot_number in snapshot_number_list]
    
    redshifts_of_snapshots = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    bin_edges = (redshifts_of_snapshots[:-1] + redshifts_of_snapshots[1:]) / 2.0
    bin_indices = np.maximum(np.digitize(redshift, bin_edges) - 1, 0)  # Ensure at least 0

    for i, snapshot_number in enumerate(snapshot_number_list):
        ### Skip if no haloes in this redshift bin
        mask_z = (bin_indices == i)
        if not np.any(mask_z):
            print(f"# No haloes in redshift bin {i} (snapshot number {snapshot_number}), skipping...")
            continue
    
        ### load Transformer
        model_dir = f"{args.model_dir}/transformer1_{snapshot_number}_use_vel{suffix_list[i]}"

        with open(f"{model_dir}/args.json", "r") as f:
            opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        print("opt: ", opt)

        norm_params = np.array(opt.norm_params)
        xmin = norm_params[:,0]
        xmax = norm_params[:,1]

        model = my_model(opt)
        model.to(device)

        model.load_state_dict(torch.load("{}/model.pth".format(model_dir)))
        model.eval()
        print(model)

        ### generate galaxies
        print(f"# Generate galaxies (batch size: {opt.batch_size})")

        max_ids = np.loadtxt(max_ids_file_list[i], dtype=int)
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

from line_model import line_dict 
line_names = list(line_dict.keys())
NLINE = len(line_names)

def load_data(path, zlim=[0,-1], print_total=True):
    print("# Load {}".format(path))
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        frequency = f["/frequency"][:] / GHz # [GHz]
        intensity = f["/total_intensity"][:]

        frequency = frequency[zlim[0]:zlim[1]]
        intensity = intensity[:, :, zlim[0]:zlim[1]]

        intensity_line = {}
        for line_name in line_names:
            dataset_name = "intensity_{}".format(line_name)
            if dataset_name in f:
                tmp = f[dataset_name][:]
                intensity_line[line_name] = tmp[:, :, zlim[0]:zlim[1]]
                if print_total and np.sum(intensity_line[line_name]) > 0:
                    print("Total intensity {}: {:e}".format(line_name, np.sum(intensity_line[line_name])))
    
    print("freqency shape: ", frequency.shape)
    print("intensity shape: ", intensity.shape)

    return frequency, intensity, intensity_line, attrs

def plot_mean(frequency, intensity, intensity_line, title=None, lines_to_show=line_names, ylim=None, nsmooth=1, logx=False):
    """
    plot mean intensity
    input: 
        frequency: (Nf+1,) frequency [GHz]
        intensity: (Nx, Ny, Nf) intensity [Jy/sr]
        intensity_line: (Nline, Nx, Ny, Nf) intensity [Jy/sr]
        title: title of the plot
        lines_to_show: list of line names to show
        ylim: y-axis limit
        nsmooth: smoothing scale
        logx: if True, use log scale for x-axis
    """

    plt.figure()
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Mean intensity [Jy/sr]")

    frequency_bin = 0.5 * (frequency[1:] + frequency[:-1])
    mean = np.mean(intensity, axis=(0, 1))
    if nsmooth > 1:
        mean = gaussian_filter(mean, sigma=nsmooth)
    plt.plot(frequency_bin, mean, color="k", label="total")

    count = 0
    for i, line_name in enumerate(line_names):
        if line_name in lines_to_show:
            if line_name in intensity_line:
                mean_line = np.mean(intensity_line[line_name], axis=(0, 1))
                if nsmooth > 1:
                    mean_line = gaussian_filter(mean_line, sigma=nsmooth)

                ls = "-" if count < 10 else "--"
                plt.plot(frequency_bin, mean_line, ls=ls, label=f"{line_name}")
                count += 1
            else:
                print(f"Warning: {line_name} not found in intensity_line, skipping...")

    if len(lines_to_show) > 4:
        plt.legend(bbox_to_anchor=(1,1.1))
    else:
        plt.legend()
    plt.title(title)
    plt.yscale("log")
    if logx:
        plt.xscale("log")
    plt.ylim(ylim)
    
def show_map(frequency, intensity, intensity_line, side_length, lines_to_show=line_names, dy=1, use_log=False, smoothing=0, noise_sigma=0):
    """
    show intensity map in x-z plane
    input:
        frequency: (Nf+1,) frequency [GHz]
        intensity: (Nx, Ny, Nf) intensity [Jy/sr]
        intensity_line: (Nline, Nx, Ny, Nf) intensity [Jy/sr]
        side_length: side length [arcsec]
        lines_to_show: list of line names to show
        use_log: if True, use log scale for intensity
        smoothing: smoothing scale
    """

    Nmap = len(lines_to_show) + 1
    if noise_sigma > 0:
        Nmap += 1


    plt.figure(figsize=(10, Nmap*2))
    plt.subplots_adjust(hspace=1)

    fmin = frequency[0]
    fmax = frequency[-1]

    def show_ticks(fmin=fmin, fmax=fmax, freq=frequency):
        if fmax - fmin < 150:
            tick_values = [ int(fmin/100) * 100 + 20*i for i in range(10) ]
        elif fmax - fmin > 10000:
            tick_values = [ int(fmin/10000) * 10000 + 2000*i for i in range(30) ]
        else:
            tick_values = [ int(fmin/100) * 100 + 100*i for i in range(10) ]
        tick_values = [v for v in tick_values if v > fmin and v < fmax]
        interpolator = interp1d(freq, np.arange(len(freq)))
        tick_positions = interpolator(tick_values)
        plt.xticks(ticks=tick_positions, labels=tick_values)

    def show_a_map(imap, map, side_length=side_length, iy=0, dy=1, vmin=None, vmax=None, label=None, use_log=use_log):
        plt.subplot2grid((Nmap,1), (imap,0))
        map_xz = map[:, iy:iy+dy, :].sum(axis=1)
        if use_log: map_xz = np.log10(map_xz)
        if smoothing > 0:
            map_xz = gaussian_filter(map_xz, sigma=smoothing)
        cmap = plt.cm.viridis
        cmap.set_bad(color=cmap(0))
        plt.imshow(map_xz, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(label)
        plt.colorbar(label="log I [Jy/sr]")
        plt.yticks([])
        x = side_length / 3600
        plt.ylabel("{:.1f} deg".format(x))
        plt.xlabel("frequency [GHz]")
        show_ticks()

    if dy < 0:
        dy = intensity.shape[1]
    print("Projection along y-axis: {:d} slices".format(dy))
        
    count = 0
    if noise_sigma > 0:
        intensity_with_noise = intensity + np.random.normal(0, noise_sigma, intensity.shape)
        show_a_map(count, intensity_with_noise, dy=dy, label="total with noise")
        count += 1

    show_a_map(count, intensity, dy=dy, label="total")
    count += 1
    for i, line_name in enumerate(line_names):
        if line_name in lines_to_show:
            if line_name in intensity_line:
                show_a_map(count, intensity_line[line_name], dy=dy, label=line_name)
                count += 1
            else:
                print(f"Warning: {line_name} not found in intensity_line, skipping...")

def my_fft(X, L=None, b=1.): # b = 2 * np.pi for inverse FT
    """
    Actually whatever dimension is ok
    input:
        X: (Nx, Ny, Nz) data 
        L: (3,) or float, box size
        b: normalization factor
    output:
        ft: (Nx, Ny, Nz) Fourier transform
        freq: (Nx, Ny, Nz) frequency
    """

    dim = len(X.shape)
    N = np.array(X.shape)

    if L is None:
        L = N
    elif np.isscalar(L):
        L = L * np.ones(dim)

    dx = np.array([float(l)/float(n) for l, n in zip(L, N)])
    Vx = np.product(dx) # volume of a cell

    freq = [fftfreq(n, d=d)*2*np.pi for n, d in zip(N, dx)]
    ft = Vx * fftn(X) * np.sqrt(1/np.abs(b)) ** dim
    
    return ft, freq

def calc_noise_power(sigma_noise_Jy_sr, freq, intensity, side_length=3600, line_name="CO(1-0)", with_hlittle=True, cosmo=cosmo_default):
       
    redshifts = line_dict[line_name][0] / GHz / freq - 1
    redshift_mean = np.mean(redshifts)
    nu_rest = line_dict[line_name][0] / GHz

    Lx = arcsec_to_cMpc(side_length, redshift_mean) # [cMpc/h]
    Ly = arcsec_to_cMpc(side_length, redshift_mean) # [cMpc/h]
    Lz = freq_to_comdis(freq[0], nu_rest) - freq_to_comdis(freq[-1], nu_rest) 

    dx = Lx / intensity.shape[0]
    dy = Ly / intensity.shape[1]
    dz = Lz / intensity.shape[2]
    
    dL = np.array([dx, dy, dz])

    hlittle = cosmo.H0.value / 100
    if with_hlittle:
        dL *= hlittle
    dV = np.prod(dL) 

    P_noise = sigma_noise_Jy_sr**2 * dV

    return P_noise


def calc_power(freq_obs, intensity, intensity2=None, side_length=3600, line_name="CO(1-0)", dlogk=0.2, with_hlittle=True, logkpara_min=-10, logkperp_min=-10, sigma_noise=0, sigma_noise2=0, cosmo=cosmo_default): 
    """
    input:
        freq: (N,) frequency [GHz]
        intensity: (Nx, Ny, Nf) intensity [arbitrary unit]
        side_length: side length [arcsec]
        line_name: line name for which the distance is calculated
        dlogk: bin width for logk
        with_hlittle: if True, multiply h
        logkpara_min: minimum value of logkpara
        logkperp_min: minimum value of logkperp
    output:
        k: (Nk,) wavenumber [h/cMpc^-1]
        power1d: (Nk,) power spectrum [input unit^2 * (cMpc/h)^3]
    """
    
    redshifts = line_dict[line_name][0] / GHz / freq_obs - 1
    redshift_mean = np.mean(redshifts)
    nu_rest = line_dict[line_name][0] / GHz

    print("Use {} rest-frame frequency".format(line_name))
    print("redshift: {:.2f} - {:.2f} (mean: {:.2f})".format(redshifts[-1],redshifts[0],redshift_mean))
    
    Lx = arcsec_to_cMpc(side_length, redshift_mean) # [cMpc]
    Ly = arcsec_to_cMpc(side_length, redshift_mean) # [cMpc]
    Lz = freq_to_comdis(freq_obs[0], nu_rest) - freq_to_comdis(freq_obs[-1], nu_rest) # [cMpc]

    ## Fourier transform
    L = np.array([Lx, Ly, Lz])
    dL = np.array([Lx / intensity.shape[0], Ly / intensity.shape[1], Lz / intensity.shape[2]])
    hlittle = cosmo.H0.value / 100
    if with_hlittle:
        L *= hlittle # [cMpc/h]
        dL *= hlittle # [cMpc/h]
    V = np.prod(L) 
    dV = np.prod(dL) # [cMpc/h]^3
    
    ft, freq = my_fft(intensity, L=L)
    if intensity2 is None:
        ft2 = ft
    else:
        ft2, freq2 = my_fft(intensity2, L=L)

    power_spectrum = np.real(ft* np.conj(ft2)) / V
    kx, ky, kz = np.meshgrid(freq[0], freq[1], freq[2], indexing='ij')

    ## noise power spectrum
    power_noise = sigma_noise**2 * dV

    ## compute angular-averaged power spectrum
    logk = np.log10(np.sqrt(kx**2 + ky**2 + kz**2))
    logkpara = np.log10(np.abs(kz))
    logkperp = np.log10(np.sqrt(kx**2 + ky**2))
    
    logk_bins = np.arange(-1.4, 1.3, dlogk)#dk=k*dlogk
    
    power1d = np.zeros(len(logk_bins) - 1)
    power1d_err = np.zeros(len(logk_bins) - 1)
    for i in range(len(logk_bins) - 1):
        mask = (logk >= logk_bins[i]) & (logk < logk_bins[i+1]) & (logkpara > logkpara_min) & (logkperp > logkperp_min)
        Nk = mask.sum()

        if np.any(mask):
            power1d[i] = np.mean(power_spectrum[mask])
            power1d_err[i] = (power1d[i] + power_noise) / np.sqrt(Nk)

    return 10**(0.5*(logk_bins[1:]+logk_bins[:-1])), power1d, power1d_err