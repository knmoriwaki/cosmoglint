import sys
import os
import numpy as np

import h5py
from cosmoglint.utils.stats import compute_power
from cosmoglint.utils import make_density_map

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="../data/generated_data")
parser.add_argument("--output_dir", type=str, default="../data/generated_data/statistics")
parser.add_argument("--fname_id", type=str, default="catalog_ddpm_quadratic_crossformer_33_npix16_ep6000_hd128_nl4_nlcond3_nlpyr0_w0.01_lr1e-4_pad_with_flag")
parser.add_argument("--dm_fname", type=str, default="TNG300-1-Dark.h5")
parser.add_argument("--stats_name", type=str, default="power")

parser.add_argument("--sfrmin", type=float, default=1)

parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--seed_end", type=int, default=1)

parser.add_argument("--npix", type=int, default=128, help="npix for power spectrum calculation")
parser.add_argument("--npix_orig", type=int, default=256)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def load_catalog_data(fname):
    #print("Load catalog data from", fname)
    with h5py.File(fname, "r") as f:
        header = {key: f.attrs[key] for key in f.attrs}
        x_list = []
        for key in ["SubhaloPos", "SubhaloSFR", "SubhaloStellarMass"]:
            if key not in f:
                print("Key '{}' not found in file".format(key))
                continue

            x = f[key][:]
            if x.ndim == 1:
                x = x[:,None]
            if key == "SubhaloVel":
                x = x[:,-1:] # vz only
            x_list.append(x)
        
        cat = np.concatenate(x_list, axis=1)

    return cat, header

pos_idx = [0,1,2]
sfr_idx = 3
mstar_idx = 4

def calc_power(cat, boxsize, fout):
    dx = boxsize / args.npix
    pos = cat[:, :3]

    nden = make_density_map(pos/dx, args.npix, mode="CIC")
    delta_nden = nden / nden.mean() - 1.0 # delta

    power, k, var = compute_power(delta_nden, boxlength=boxsize, nbins=20, log_bins=True, verbose=False)
    
    k_values = 0.5 * (k[1:] + k[:-1])
    power_shot = np.ones_like(k_values) * boxsize**3 / nden.sum()

    with open(fout, "w") as f:
        for i in range(len(k_values)):
            f.write("{:e} {:e} {:e} {:e}\n".format(k_values[i], power[i], var[i], power_shot[i]))
    print("Saved power spectrum to", fout)

def calc_lim_power(cat, boxsize, fout):
    dx = boxsize / args.npix
    pos = cat[:, pos_idx]

    intensity = make_density_map(pos/dx, args.npix, weight=cat[:,sfr_idx], mode="CIC")

    power, k, var = compute_power(intensity, boxlength=boxsize, nbins=20, log_bins=True, verbose=False)
    
    k_values = 0.5 * (k[1:] + k[:-1])
    power_shot = np.ones_like(k_values) * boxsize**3 / (cat[:,sfr_idx]**2).sum()

    with open(fout, "w") as f:
        for i in range(len(k_values)):
            f.write("{:e} {:e} {:e} {:e}\n".format(k_values[i], power[i], var[i], power_shot[i]))
    print("Saved lim power spectrum to", fout)

def calc_cross_power(cat, density_map, boxsize, fout):
    dx = boxsize / args.npix
    pos = cat[:, pos_idx] 

    nden = make_density_map(pos/dx, args.npix, mode="CIC")
    delta_nden = nden / nden.mean() - 1.0 # delta

    power, k, var = compute_power(delta_nden, density_map / density_map.mean() - 1.0, boxlength=boxsize, nbins=20, log_bins=True, verbose=False)

    k_values = 0.5 * (k[1:] + k[:-1])

    with open(fout, "w") as f:
        for i in range(len(k_values)):
            f.write("{:e} {:e} {:e}\n".format(k_values[i], power[i], var[i]))
    print("Saved cross power spectrum to", fout)


def calc_lim_cross_power(cat, density_map, boxsize, fout):
    dx = boxsize / args.npix
    pos = cat[:, pos_idx]

    intensity = make_density_map(pos/dx, args.npix, weight=cat[:,sfr_idx], mode="CIC")

    power, k, var = compute_power(intensity, density_map / density_map.mean() - 1.0, boxlength=boxsize, nbins=20, log_bins=True, verbose=False)
    
    k_values = 0.5 * (k[1:] + k[:-1])
    
    with open(fout, "w") as f:
        for i in range(len(k_values)):
            f.write("{:e} {:e} {:e}\n".format(k_values[i], power[i], var[i]))
    print("Saved lim power spectrum to", fout)


def calc_r_profile(cat, boxsize, fout):
    from cosmoglint.utils.stats import radial_profile_around_points

    r_min = 3.0
    r_max = 50.0  
    nbins = 20

    pos = cat[:,pos_idx] # [Mpc]
    voxel_size = boxsize / npix_to_show # [Mpc]

    profiles, radii = radial_profile_around_points(density_map, pos, r_min, r_max, nbins, voxel_size=voxel_size)

    print("radii shape:", radii.shape)        # (nbins,)
    print("profiles shape:", profiles.shape)  # (L, nbins)
    print(profiles[0])

    # Print out
    mask = ~np.isnan(profiles).any(axis=1)
    p = profiles[mask]
    if len(p) == 0:
        print("No valid data")
        sys.exit(1)

    p_mean = p.mean(axis=0)
    p_var = p.var(axis=0)
    with open(fout, "w") as f:
        for i in range(len(radii)):
            f.write("{:e} {:e} {:e}\n".format(radii[i], p_mean[i], p_var[i]))
    print("Saved radial profile data to ", fout)

### Load data
snapshot_number = 33
sim_name = "TNG300-3-Dark"

with h5py.File(args.dm_fname, "r") as f:
    density_map = f["density_map"][:] 
npix_to_show = 64
if "npix16" in args.fname_id:
    npix_to_show = 128
if "npix32" in args.fname_id:
    npix_to_show = 256

npix_to_show = args.npix

density_map = density_map[-npix_to_show:,-npix_to_show:,-npix_to_show:]

if args.fname_id == "TNG":
    file_name_list_cat = ["../data/mesh/TNG300-1/TNG300-1_33.h5"]
else:
    file_name_list_cat = [ "{}/{}.seed{:d}.h5".format(args.base_dir, args.fname_id, seed) for seed in range(args.seed_start, args.seed_end) ]

for i_cat, file_name in enumerate(file_name_list_cat):

    if not os.path.exists(file_name):
        print("File {} does not exsits. break.".format(file_name))
        break

    cat, header = load_catalog_data(file_name)        
    if args.fname_id== "TNG":
        xmin = 205000 * (args.npix_orig - npix_to_show) / args.npix_orig
        mask = (cat[:,pos_idx[0]]>xmin) & (cat[:,pos_idx[1]] > xmin) & (cat[:,pos_idx[2]] > xmin)
        cat = cat[mask]
        cat[:,pos_idx] -= xmin
        header["BoxSize"] *= npix_to_show / args.npix_orig
    else:
        npix_to_use =  header["npix_to_use"]
        xmin = header["BoxSize"] * (npix_to_use - npix_to_show) / npix_to_use
        mask = (cat[:,pos_idx[0]]>xmin) & (cat[:,pos_idx[1]] > xmin) & (cat[:,pos_idx[2]] > xmin)
        cat = cat[mask]
        cat[:,pos_idx] -= xmin
        header["BoxSize"] *= npix_to_show / npix_to_use

    cat = np.random.permutation(cat)
    mask = cat[:,sfr_idx] > args.sfrmin
    cat = cat[mask]

    cat[:,pos_idx] = cat[:,pos_idx] / 1e3
    boxsize = header["BoxSize"] / 1e3

    if args.fname_id == "TNG":
        if args.npix == npix_to_show:
            fout = "{}/{}_sfrmin{}_TNG.txt".format(args.output_dir, args.stats_name, str(args.sfrmin))    
        else:
            fout = "{}/power_sfrmin{}_npix{:d}_TNG.txt".format(args.output_dir, str(args.sfrmin), args.npix)
    else:
        if args.npix == npix_to_show:
            fout = "{}/{}_sfrmin{}_{}.seed{:d}.txt".format(args.output_dir, args.stats_name, str(args.sfrmin), args.fname_id, i_cat)
        else:
            fout = "{}/{}_sfrmin{}_npix{:d}_{}.seed{:d}.txt".format(args.output_dir, args.stats_name, str(args.sfrmin), args.npix, args.fname_id, i_cat)
            
    if ("r_profile" in fout) and os.path.exists(fout):
        print("{} exists. Skip".format(fout))
        break 
    
    if args.stats_name == "power":
        calc_power(cat, boxsize, fout)
    elif args.stats_name == "cross_power":
        calc_cross_power(cat, density_map, boxsize, fout)
    elif args.stats_name == "r_profile":
        calc_r_profile(cat, boxsize, fout)
    elif args.stats_name == "lim_power":
        calc_lim_power(cat, boxsize, fout)
    elif args.stats_name == "lim_cross_power":
        calc_lim_cross_power(cat, density_map, boxsize, fout)
    elif args.stats_name == "corr":
        from map2points.utils.power import calc_corr
        r, xi, varxi = calc_corr(cat[:,pos_idx], boxsize, n_random=10*len(cat), fout=fout)
        print("Saved correlation function to", fout)
    else:
        NotImplementedError("{} is not implemented".format(args.stats_name))



