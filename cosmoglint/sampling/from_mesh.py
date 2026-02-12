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

from cosmoglint.utils import normalize, get_index_list
from cosmoglint.datasets.dataset_mesh import get_new_region, round_offsets

def sample_galaxies_from_mesh(
        x_in, 
        model, 
        opt,  
        device='cpu',
        **kwargs
    ): 

    B = opt.batch_size
    x_shape = (opt.max_length, opt.num_features_in)
    npix_patch = opt.npix_patch

    pos_idx = get_index_list(opt.output_features, "SubhaloPos")
    
    overlap = 8 # overlap between patches [pixels]    
    overlap_half = overlap // 2

    nx, ny, nz, _ = x_in.shape
    stride = npix_patch - overlap

    patches = []
    patch_start_indices = []
    patch_valid_ranges = []

    def get_valid_range(start, nmax, prev_valid_end):
        start = min(start, nmax - npix_patch)
        end = start + npix_patch

        valid_start = prev_valid_end
        #valid_start = start + (0 if start == 0 else overlap_half) 
        valid_end = nmax if end == nmax else end - overlap_half

        #if end == nmax:
        #    valid_start = prev_valid_end
            
        return start, end, valid_start, valid_end

    valid_x_end = 0
    for ix in range(0, nx - overlap, stride):
        start_x, end_x, valid_x_start, valid_x_end = get_valid_range(ix, nx, valid_x_end)
                
        valid_y_end = 0
        for iy in range(0, ny - overlap, stride):
            start_y, end_y, valid_y_start, valid_y_end = get_valid_range(iy, ny, valid_y_end)
                
            valid_z_end = 0
            for iz in range(0, nz - overlap, stride):
                start_z, end_z, valid_z_start, valid_z_end = get_valid_range(iz, nz, valid_z_end)

                x_patch = x_in[start_x:end_x, start_y:end_y, start_z:end_z]
                
                patches.append(x_patch)
                patch_start_indices.append((start_x, start_y, start_z))
                patch_valid_ranges.append(((valid_x_start, valid_x_end), (valid_y_start, valid_y_end), (valid_z_start, valid_z_end)))

    patches = torch.stack(patches).float() # (num_patches, npix_patch, npix_patch, npix_patch, 1)
    patches = patches.permute(0, 4, 1, 2, 3) # (num_patches, 1, npix_patch, npix_patch, npix_patch)

    num_batch = (len(patches) + B - 1) // B
    generated = []
    for batch_idx in tqdm(range(num_batch)):
        start = batch_idx * B
        end = min(start + B, len(patches))
        patches_batch = patches[start:end].to(device) # (B, npix_patch, npix_patch, npix_patch)
        cond = {
            "mesh3d": patches_batch,
        }
        with torch.no_grad():
            generated_batch, _ = model.generate(cond, **kwargs)

        if opt.use_flat_representation:
            generated_batch.reshape(len(generated_batch), -1, opt.num_features_in)
        
        ### Use non-negative positions and sfr; negative values represent invalid galaxies in training
        mask = (generated_batch >= 0).all(dim=-1) # (B, seq_length)

        ### Convert pos to global idx
        start_indices = torch.tensor(patch_start_indices[start:end], device=device).unsqueeze(1) # (B, 1, 3)
        generated_batch[..., pos_idx] = generated_batch[...,pos_idx] * npix_patch + start_indices # convert [0, 1) -> [start_index, start_index + npix_patch)

        ### Use positions within the valid ranges
        valid_ranges = torch.tensor(patch_valid_ranges[start:end], device=device) # (B, 3, 2)
        valid_min = valid_ranges[:,:,0].unsqueeze(1) # (B, 3)
        valid_max = valid_ranges[:,:,1].unsqueeze(1) # (B, 3)
        valid_mask = ( (generated_batch[...,pos_idx] >= valid_min) & (generated_batch[...,pos_idx] < valid_max) ).all(dim=-1) # (B, seq_length)

        ### Apply masks
        mask = valid_mask & mask # (B, seq_length)
        masked_output = generated_batch[mask] # (num_gal_batch, num_features_out)
        generated.append(masked_output)
            
    generated = torch.cat(generated, dim=0).cpu().numpy() # (num_gal, num_features_out)

    return generated

def generate_bboxes(round_id, npix_patch, cond_shape=(512,512,512)):
    """
    Input: 
        round_id: int
        npix_patch: int
        npix: tuple (3,)
    Output: 
        bboxes: np.ndarray (N,3), start indices of N bboxes
    """
    hx, hy, hz = round_offsets(round_id, npix_patch)
    sx = np.arange(hx, cond_shape[0] - npix_patch + 1 - hx, npix_patch, dtype=int)
    sy = np.arange(hy, cond_shape[1] - npix_patch + 1 - hy, npix_patch, dtype=int)
    sz = np.arange(hz, cond_shape[2] - npix_patch + 1 - hz, npix_patch, dtype=int)

    g = np.meshgrid(sx, sy, sz, indexing='ij')
    bboxes = np.stack(g, axis=-1).reshape(-1, 3)

    return bboxes

def make_ctx(
        generated, 
        round_id=0, 
        cond_shape=(512,512,512), 
        npix_patch=32, 
        x_shape=(10,3), 
        pos_idx=[0,1,2], 
        sort=True, 
        device='cpu'
    ):
    """
    Input:
        generated: torch.tensor (num_gal, num_features_in)
        round_id: int
        npix: tuple
        npix_patch: int
        x_shape: tuple
        fill_value: float
        device: str
    Output: 
        generated_kept: torch.tensor (N, num_features_in)
        ctx: torch.tensor (N', num_features_in)
    """
    
    dtype = torch.float32
    hx, hy, hz = round_offsets(round_id, npix_patch)
    max_length = x_shape[0]
    
    n_axis = [ p // npix_patch for p in cond_shape[:3] ]
    nx = n_axis[0] - (0 if hx == 0 else 1)
    ny = n_axis[1] - (0 if hy == 0 else 1)
    nz = n_axis[2] - (0 if hz == 0 else 1)

    B = nx * ny * nz 

    # Initialize ctx
    ctx = torch.zeros((B, *x_shape), dtype=dtype, device=device)
    mask_ctx = torch.zeros((B, *x_shape), dtype=torch.bool, device=device)

    if generated is None:
        generated_kept = None

    else:
        # Convert global pos to round-dependent local pos 
        lo = torch.tensor([hx, hy, hz], device=device, dtype=dtype)
        hi = torch.tensor([cond_shape[0] - hx, cond_shape[1] - hy, cond_shape[2] - hz], device=device, dtype=dtype)

        pos = generated[:, pos_idx] # (num_gal, 3)
        valid = ((pos >= lo) & (pos < hi)).all(dim=1) # (num_gal,) -- galaxies whitin any bboxes in this round

        ijk = torch.floor( (pos - lo) / float(npix_patch)).long() # (num_gal, 3)
        patch_id = ijk[:,0] * (ny*nz) + ijk[:,1] * nz + ijk[:,2] # (num_gal,)

        starts = ijk * npix_patch + lo # (num_gal, 3)
        pos_local = ( pos - starts ) / npix_patch # (num_gal, 3) This is in [0, 1)

        # Remove galaxies in new regions
        new_region_lo, new_region_hi = get_new_region(round_id, device=device) # (3,) -- [-1, 1)
        in_new_region = ((pos_local >= new_region_lo[None,:]) & (pos_local < new_region_hi[None,:])).all(dim=1) # (num_gal,) 
        
        # Keep galaxies outside this round's bboxes and those in bboxes but not in new regions
        kept = (~valid) | (valid & ~in_new_region) 
        generated_kept = generated[kept] # (num_gal', ) 
        pos_local = pos_local[kept]
        patch_id = patch_id[kept]
        
        valid = valid[kept]

        print("{:d} (out of {:d}) galaxies were discarded.".format(len(kept)-len(generated_kept), len(kept)))

        # make context 
        pid_for_ctx = patch_id[valid]
        ctx_candidates = generated_kept[valid]
        ctx_candidates[:, pos_idx] = pos_local[valid]

        lengths = torch.bincount(pid_for_ctx, minlength=B) # (B, )
        offs = torch.nn.functional.pad(torch.cumsum(lengths, 0), (1,0))  # (B+1,)
        
        order = torch.argsort(pid_for_ctx)
        pts_sorted = ctx_candidates[order]
        
        for b in range(B):
            length = int(lengths[b].item())
            if length:
                start, end = int(offs[b].item()), int(offs[b+1].item())
                ctx_now = pts_sorted[start:end]
                if sort:
                    _, indices = torch.sort(ctx_now[:, 0])
                    ctx_now = ctx_now[indices]
                
                ctx[b, :length] = ctx_now[:max_length] 
                mask_ctx[b, :length] = True

    return generated_kept, ctx, mask_ctx

def sample_galaxies_from_mesh_continuous(
        x_in, 
        model, 
        opt,
        num_rounds=8, 
        device='cpu',
        **kwargs
    ):

    B = opt.batch_size

    x_shape = (opt.max_length, opt.num_features_in)

    npix_patch = opt.npix_patch

    pos_idx = get_index_list(opt.output_features, "SubhaloPos")
    
    generated_kept = None
    for round_id in range(num_rounds):
        new_region_lo, new_region_hi = get_new_region(round_id, device=device)
        
        generated_kept, ctx, mask_ctx = make_ctx(generated_kept, 
                                       round_id = round_id, 
                                       cond_shape = x_in.shape, 
                                       npix_patch = npix_patch, 
                                       x_shape = x_shape, 
                                       pos_idx = pos_idx,
                                       device = device
                                       ) 
        # generated_kept: (num_gal, num_features_in), ctx: (num_boxes, max_lenght, num_features_in)
        
        if opt.use_flat_representation:
            ctx = ctx.reshape(len(ctx), -1, 1)
            mask_ctx = mask_ctx.reshape(len(mask_ctx), -1, 1)

        start_indices = generate_bboxes(round_id, npix_patch, x_in.shape)

        num_batch = (len(start_indices) + B - 1) // B

        generated_new = []
        for batch_idx in tqdm(range(num_batch), desc="Round {:d}".format(round_id)):
            start = batch_idx * B
            end = min(start + B, len(start_indices))

            lo = new_region_lo.unsqueeze(0).expand(end-start, -1)
            hi = new_region_hi.unsqueeze(0).expand(end-start, -1)

            start_indices_batch = start_indices[start:end]
            rng = np.arange(npix_patch, dtype=np.int64)
            ix = (start_indices_batch[:, [0]] + rng[None, :])[:, :, None, None]  # (B, npix_patch, 1, 1)
            iy = (start_indices_batch[:, [1]] + rng[None, :])[:, None, :, None]  # (B, 1, npix_patch, 1)
            iz = (start_indices_batch[:, [2]] + rng[None, :])[:, None, None, :]  # (B, 1, 1, npix_patch)
            
            x_batch = x_in[ix, iy, iz].to(torch.float32).to(device) # (B, npix_patch, npix_patch, npix_patch, num_features_cond)
            x_batch = x_batch.permute(0, 4, 1, 2, 3) # (B, num_features_cond, npix_patch, npix_patch, npix_patch)

            cond = {
                "mesh3d": x_batch, 
                "context": ctx[start:end],
                "mask_ctx": mask_ctx[start:end],
                "boundary": torch.cat([lo, hi], dim=-1)
            }

            with torch.no_grad():
                generated_batch, _ = model.generate(cond, **kwargs)

            if opt.use_flat_representation:
                generated_batch = generated_batch.reshape(len(generated_batch), -1, opt.num_features_in) # (B, max_length, num_features_in)

            # Using galaxies in new region and valid parameter space only
            # Selection of those in new region should be before the conversion to [0,1)
            in_new_region = ((generated_batch[...,pos_idx] >= lo[:,None,:]) & (generated_batch[...,pos_idx] < hi[:,None,:])).all(dim=-1) # (B, seq_length)
            
            # Convert [0, 1) to global [start_index, start_index + npix_patch)
            starts_t = torch.from_numpy(start_indices_batch).to(device=device, dtype=generated_batch.dtype) # tensor
            generated_batch[..., pos_idx] = generated_batch[..., pos_idx] * npix_patch + starts_t[:,None,:]

            # Apply mask and append
            generated_new.append(generated_batch[in_new_region])

        generated_new = torch.cat(generated_new, dim=0) # (num_gal_new, num_features_in)
        print("{:d} galaxies were newly generated in Round {:d}.".format(len(generated_new), round_id))
        
        if generated_kept is None:
            generated_kept = generated_new
        else:
            generated_kept = torch.cat([generated_kept, generated_new], dim=0)

    generated_kept = generated_kept.cpu().numpy() # (num_gal_tot, num_features_in)

    return generated_kept