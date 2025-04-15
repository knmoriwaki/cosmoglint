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

parser.add_argument("--model_dir", type=str, default=None, help="The directory of the model. If not given, use 4th column as intensity.")
parser.add_argument("--NN_model_dir", type=str, default=None, help="The directory of the NN model. If not given, use 4th column as intensity.") 

args = parser.parse_args()

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
    if "old_version" in args.input_fname:
        M, theta, phi, _, redshift = load_old_plc(args.input_fname)
        logm = np.log10(M)
    else:
        import Pinocchio.scripts.ReadPinocchio5 as rp
        myplc = rp.plc(args.input_fname)
        
        logm = np.log10( myplc.Mass )
        theta = myplc.data["theta"]
        phi = myplc.data["phi"]
        redshift = myplc.data["obsz"]

    print(redshift)
    sys.exit(0)

else:
    raise ValueError("Unknown input file format")

print(f"# Redshift: {redshift}")
H = cosmo.H(redshift).to(u.km/u.s/u.Mpc).value #[km/s/Mpc]
hlittle = cosmo.H(0).to(u.km/u.s/u.Mpc).value / 100.0 
a = 1 / (1 + redshift)

### Assign intensity
intensity = np.zeros((args.npix, args.npix, args.npix_z))

if args.model_dir == None:
    # Use original values in simulation data (4th column)
    value = data[:,7]

    if args.redshift_space:
        print(f"# Map in redshift space")
        # Convert z -> z + vz/aH
        pos[:,2] += vel[:,2] / a / H * hlittle # [Mpc/h]

    for i in range(len(pos)):
        ix = np.array([pos[i,0], pos[i,1], pos[i,2]]) / dx_pix

        if value[i] < -1:
            continue
        if any(ix < 0) or any(ix >= npix):
            continue
        intensity[int(ix[0]), int(ix[1]), int(ix[2])] += 10** value[i]

else:
    from Transformer.model import my_model
    from Transformer.NN.model import my_NN_model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_params = np.loadtxt("./Transformer/norm_params.txt")
    xmin = norm_params[:,0]
    xmax = norm_params[:,1]

    ### load Transformer
    with open(f"{args.model_dir}/args.json", "r") as f:
        opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt: ", opt)

    model = my_model(opt)
    model.to(device)

    model.load_state_dict(torch.load(f"{args.model_dir}/model.pth"))
    model.eval()
    print(model)

    ### load NN
    with open(f"{args.NN_model_dir}/args.json", "r") as f:
        opt_NN = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    print("opt_NN: ", opt_NN)

    model_NN = my_NN_model(opt_NN)
    model_NN.to(device)

    model_NN.load_state_dict(torch.load(f"{args.NN_model_dir}/model.pth"))
    model_NN.eval()
    print(model_NN)

    ### normalize data
    logm = (logm - xmin[0]) / (xmax[0] - xmin[0])
    logm = torch.from_numpy(logm).float().to(device)

    threshold = 1. / opt.num_features_out
    args.threshold = threshold
    threshold = 10 ** ( threshold * (xmax[1] - xmin[1]) + xmin[1] )

    num_batch = (len(logm) + opt.batch_size - 1) // opt.batch_size
    #for batch_idx in range(num_batch):
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * opt.batch_size 

        logm_batch = logm[start: start+opt.batch_size] # (batch_size, 1)
        generated, _ = model.generate(logm_batch) # (batch_size, seq_length, 1)
        generated_npy = generated.cpu().detach().numpy()

        for i, (m, g) in enumerate(zip(logm_batch, generated)):
            # m: (1, )
            # g: (seq_length, 1)

            i_central = i + start # index of the central galaxy

            x_NN = torch.cat([m.expand(len(g), 1), g], dim=1) # (seq_length, 2)
            g_NN, _ = model_NN.generate(x_NN) # (seq_length, 2)

            g = g.cpu().detach().numpy()
            g_NN = g_NN.cpu().detach().numpy()

            sfr = g[:,0] * (xmax[1] - xmin[1]) + xmin[1] 
            distance = g_NN[:,0] * (xmax[2] - xmin[2]) + xmin[2] 
            vr = g_NN[:,1] * (xmax[3] - xmin[3]) + xmin[3]
            
            sfr = 10 ** sfr
            distance = 10 ** distance
            vr = np.sign(vr) * (10 ** np.abs(vr) - 1) 
            
            for j, (s, d, v) in enumerate(zip(sfr, distance, vr)):
                
                x_now = np.zeros(3)  

                if j == 0: # central galaxy
                    x_now = pos[i_central]

                else: # satellite
                    if s < threshold: # stop if SFR is below the threshold for satellites
                        break

                    phi = np.random.uniform(0, 2 * np.pi)
                    cos_theta = np.random.uniform(-1, 1)
                    sin_theta = np.sqrt(1 - cos_theta ** 2)
                    x_now[0] = pos[i_central,0] + d * sin_theta * np.cos(phi)
                    x_now[1] = pos[i_central,1] + d * sin_theta * np.sin(phi)
                    x_now[2] = pos[i_central,2] + d * cos_theta

                    if args.redshift_space:
                        vz_gal = vel[i_central,2] - v * cos_theta
                        x_now[2] += vz_gal / a / H * hlittle 

                ix = x_now / dx_pix
                
                if any(ix < 0) or any(ix >= npix):
                    continue

                intensity[int(ix[0]), int(ix[1]), int(ix[2])] += s

args_dict = vars(args)
args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
    
with h5py.File(args.output_fname, 'w') as f:
    f.create_dataset('intensity', data=intensity)
    
    for key, value in args_dict.items():
        f.attrs[key] = value

print(f"Data cube saved as {args.output_fname}")

    