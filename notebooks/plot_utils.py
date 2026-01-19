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

import warnings, os

def short_formatwarning(msg, category, filename, lineno, line=None):
    return f"{os.path.basename(filename)}:{lineno}: {category.__name__}: {msg}\n"

warnings.formatwarning = short_formatwarning
warnings.filterwarnings("always", category=RuntimeWarning)

import astropy.units as u
from astropy.constants import c as cspeed # [m/s]
from astropy.cosmology import FlatLambdaCDM
cosmo_default = FlatLambdaCDM(H0=67.74, Om0=0.3089)

micron = 1e-4 # [cm]
GHz = 1e9
Jy = 1.0e-23        # jansky (erg/s/cm2/Hz)
arcsec = 4.848136811094e-6 # [rad] ... arcmin / 60 //
angstrom = 1e-8 # [cm]
cspeed_value = cspeed.value * 100 # [cm/s]

line_dict = {
    "CO(1-0)": [115.271 * GHz, 2601.7 * micron],
    "CO(2-1)": [230.538 * GHz, 1300.9 * micron],
    "CO(3-2)": [345.796 * GHz, 867.3 * micron],
    "CO(4-3)": [461.041 * GHz, 650.5 * micron],
    "CO(5-4)": [576.268 * GHz, 521.0 * micron],
    "CO(6-5)": [691.473 * GHz, 433.7 * micron],
    "CO(7-6)": [806.652 * GHz, 371.8 * micron],
    "CO(8-7)": [921.800 * GHz, 325.0 * micron],
    "CO(9-8)": [1036.912 * GHz, 289.0 * micron],
    "CO(10-9)": [1151.985 * GHz, 260.0 * micron],
    "CO(11-10)": [1267.014 * GHz, 237.0 * micron],
    "CO(12-11)": [1381.995 * GHz, 217.0 * micron],
    "CO(13-12)": [1496.922 * GHz, 200.0 * micron],
    "[CII]158": [1900.537 * GHz, 158.0 * micron],
    "[OIII]88": [3393.006 * GHz, 88.0 * micron],
    "[NII]205": [1461.131 * GHz, 205.0 * micron],
    "[NII]122": [2459.381 * GHz, 122.0 * micron],
    "[CI](1-0)": [492.16065 * GHz, 609.14 * micron],
    "[CI](2-1)": [809.34197 * GHz, 370.42 * micron],
    "[OIII]5007": [5.997e5 * GHz, 5007.0 * angstrom],
    "Ha": [4.568e5 * GHz, 6562.8 * angstrom],  
}

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

def load_data(path, line_names=[], zlim=[0,-1], print_total=True):
    print("# Load {}".format(path))
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        frequency = f["/frequency"][:] / GHz # [GHz]
        intensity = f["/total_intensity"][:]

        frequency = frequency[zlim[0]:zlim[1]]
        intensity = intensity[:, :, zlim[0]:zlim[1]]

        intensity_line = {}
        line_names = []
        for key in f.keys():
            if key.startswith("intensity_"):
                line_name = key[len("intensity_"):]
                line_names.append(line_name)

                tmp = f[key][:]
                intensity_line[line_name] = tmp[:, :, zlim[0]:zlim[1]]
                if print_total and np.sum(intensity_line[line_name]) > 0:
                    print("Total intensity {}: {:e}".format(line_name, np.sum(intensity_line[line_name])))
    
    print("freqency shape: ", frequency.shape)
    print("intensity shape: ", intensity.shape)

    return frequency, intensity, intensity_line, line_names, attrs

def plot_mean(frequency, intensity, intensity_line, title=None, lines_to_show=[], ylim=None, nsmooth=1, logx=False):
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
    for i, line_name in enumerate(lines_to_show):
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
    
def show_map(frequency, intensity, intensity_line, side_length, lines_to_show=[], dy=1, log_scale=False, smoothing=0, noise_sigma=0, inverse_x=False):
    """
    show intensity map in x-z plane
    input:
        frequency: (Nf+1,) frequency [GHz]
        intensity: (Nx, Ny, Nf) intensity [Jy/sr]
        intensity_line: (Nline, Nx, Ny, Nf) intensity [Jy/sr]
        side_length: side length [arcsec]
        lines_to_show: list of line names to show
        log_scale: if True, use log scale for intensity
        smoothing: smoothing scale
    """

    Nmap = len(lines_to_show) + 1
    if noise_sigma > 0:
        Nmap += 1

    if inverse_x:
        frequency = frequency[::-1]
        intensity = intensity[:, :, ::-1]
        for line_name in intensity_line:
            intensity_line[line_name] = intensity_line[line_name][:, :, ::-1]

    default_xticks_mode = "wavelength_um" if inverse_x else "frequency"

    plt.figure(figsize=(10, Nmap*2))
    plt.subplots_adjust(hspace=1)

    def show_ticks(xvalues, xleft=None, xright=None, mode="frequency"):
        if xleft is None: xleft = xvalues[0]
        if xright is None: xright = xvalues[-1]

        xmax = np.max(xvalues)
        xmin = np.min(xvalues)
        
        if mode == "frequency":
            if xmax - xmin < 150:
                tick_values = [ int(xmin/100) * 100 + 20*i for i in range(10) ]
            elif xmax - xmin < 10000:
                tick_values = [ int(xmin/100) * 100 + 100*i for i in range(10) ]
            elif xmax - xmin < 100000:
                tick_values = [ int(xmin/10000) * 10000 + 2000*i for i in range(30) ]
            else:
                tick_values = [ int(xmin/100000) * 100000 + 20000*i for i in range(30) ]
        elif mode == "wavelength_um":
            tick_values = [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        elif "redshift" in mode:
            tick_values = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        tick_values = [v for v in tick_values if v > xmin and v < xmax]
        interpolator = interp1d(xvalues, np.arange(len(xvalues)))
        tick_positions = interpolator(tick_values)
        plt.xticks(ticks=tick_positions, labels=tick_values)

    def show_a_map(imap, map, side_length=side_length, iy=0, dy=1, vmin=None, vmax=None, label=None, log_scale=log_scale, xticks_mode=default_xticks_mode):
        plt.subplot2grid((Nmap,1), (imap,0))
        map_xz = map[:, iy:iy+dy, :].sum(axis=1)
        if log_scale: map_xz = np.log10(map_xz)
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

        if xticks_mode == "frequency":
            xvalues = frequency
            plt.xlabel("frequency [GHz]")
        elif xticks_mode == "wavelength_um":
            xvalues = cspeed_value / (frequency * GHz) / micron  # [micron]
            plt.xlabel(r"wavelength $[\rm \mu m]$")
        elif "redshift" in xticks_mode:
            line_name = xticks_mode.split("_")[1]
            freq_rest = line_dict[line_name][0] / GHz
            xvalues = freq_rest / frequency - 1
            plt.xlabel("redshift")
        else:
            raise ValueError("Unknown xticks_mode: {}".format(xticks_mode))
        show_ticks(xvalues=xvalues, mode=xticks_mode)


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
    for i, line_name in enumerate(lines_to_show):
        if line_name in intensity_line:
            show_a_map(count, intensity_line[line_name], dy=dy, label=line_name, xticks_mode="redshift_{}".format(line_name))
            count += 1
        else:
            print(f"Warning: {line_name} not found in intensity_line, skipping...")

def plot_logy_with_sign(x, y, ax, pow_ymin=-2, logy_pos=None, color="k", ls="solid", lw=1.5, label="", alpha=1):
    ax.set_yscale("linear")

    if logy_pos is not None:
        logy_pos_char = ["{"+str(b)+"}" for b in logy_pos]
        labels = [r"$-10^{}$".format(b) for b in reversed(logy_pos_char)] + [r"$0$"] + [r"$10^{}$".format(b) for b in logy_pos_char ]
        ticks = [ - b + pow_ymin for b in reversed(logy_pos) ] + [0] + [ b - pow_ymin for b in logy_pos ]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    _y = []
    for b in y:
        if b > 10**pow_ymin:
            _y.append( np.log10( b ) - pow_ymin )
        elif b < -10**pow_ymin:
            _y.append( - np.log10( -b ) + pow_ymin )
        else:
            _y.append(0)
    ax.plot(x, _y, ls = ls, color=color, lw=lw, alpha=alpha, label=label)
    
def calc_ecp(x_true, y_true, alpha, sample_func, y_ref=None, Ngen=1000, show_pbar=True):
    Nsim = len(x_true)
    Nalpha = len(alpha)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_ref, torch.Tensor):
        y_ref = y_ref.detach().cpu().numpy()

    ecp = np.zeros(Nalpha)
    i_range = tqdm(range(Nsim)) if show_pbar else range(Nsim)
    for i in i_range:

        tmp_x = x_true[i].unsqueeze(0).expand(Ngen,-1)
        
        theta_ij = sample_func(tmp_x) # generated (Ngen, seq_length, num_params)
        if isinstance(theta_ij, torch.Tensor):
            theta_ij = theta_ij.detach().cpu().numpy()
            
        theta_i = y_true[i] # true (seq_length, num_params)
        theta_r = y_ref[i] # reference point (seq_length, num_params)

        d1 = np.sum((theta_ij - theta_r)**2, axis=(1, 2)) # (Ngen, )
        d2 = np.sum((theta_i - theta_r)**2) # scalar

        fi = (d1 < d2).mean()
        ecp += ( fi < 1 - alpha ).astype(int)

    ecp /= Nsim
    return ecp
