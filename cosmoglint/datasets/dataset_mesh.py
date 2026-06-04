import sys
import os
from argparse import Namespace
import random
import numpy as np

import h5py

import torch

from tqdm import tqdm

from torch.utils.data import Dataset

from cosmoglint.utils.io_utils import load_mesh_data, load_galaxy_data
from cosmoglint.utils import get_index_list
    
def get_random_patches(
    dm_density, 
    gal_data, 
    num_patches, 
    npix_patch, 
    pixel_size,
    output_features,
    max_length=10, 
    #y_cols=None,
    sort=True, 
    exclude_ratio=0,
    use_excluded_region=False,
    show_pbar=True
):
    """
    dm_density : (npix, npix, npix)
    gal_data : (N, num_params) where N is the number of galaxies in the patch
            gal_data[:,0:3] : position (N, 3) unnormalized position
            gal_data[:,3:6] : velocity (N, 3) [km/s] 
            gal_data[:,6:] : other parameters (if any)
            positions and velocities will be rotated 
            The last column is the primary parameter (e.g., SFR) to sort by.
    num_patches : Number of patches to load.
    npix_patch : Size of the patch in pixels (npix_patch, npix_patch, npix_patch)
    pixel_size : Size of a pixel in kpc/h
    max_length : Maximum number of galaxies in the patch.
    y_cols : Columns to return in the output (default: [0, 1, 2, 6] for x, y, z, primary parameter)
    sort : Whether to sort the galaxies in the patch by the primary parameter (e.g., SFR).
    npix_exclude : Number of pixels to exclude from the edges of the cube.
    """

    #if y_cols is not None:
    #    y_cols = np.array(y_cols) 

    npix = dm_density.shape[0] 
    pos_idx = get_index_list(output_features, "SubhaloPos")
    i_pos = gal_data[:, pos_idx] / pixel_size # (N, 3) pixel indices

    npix_exclude = int( npix * exclude_ratio )
    if use_excluded_region:
        if npix_exclude < npix_patch:
            raise ValueError("npix_exclude should be larger than npix_patch.")
        print("# Use excluded region")
    else:
        if npix_exclude > 0:
            print("# Exclude the corner of size ({:d})^3".format(npix_exclude))
            print("# The excluded region is {:.2f} % of the entier volume".format(100.0 * (npix_exclude/npix)**3))
        else:
            print("# Use the entier volume")

    x_list = []
    y_list = []
    if show_pbar:
        iterator = tqdm(range(num_patches), desc="Loading data")
    else:
        iterator = range(num_patches)
    for _ in iterator:

        # Select a patch randomly
        if use_excluded_region and npix_exclude > 0:
            start = np.random.randint(npix - npix_exclude, npix - npix_patch, size=3)
            end = start + npix_patch
        else:
            while True:
                start = np.random.randint(0, npix - npix_patch, size=3)
                end = start + npix_patch
                if (npix_exclude <= 0) or ( end < npix - npix_exclude ).any():
                    break

        ix, iy, iz = start
        dm_density_patch = dm_density[ix:ix+npix_patch, iy:iy+npix_patch, iz:iz+npix_patch] # (npix_patch, npix_patch, npix_patch)
        
        mask = ((i_pos >= start) & (i_pos < end)).all(axis=1)  # (N, 3)
        gal_patch = gal_data[mask] # (N, num_params) where N' is the number of galaxies in the patch
        gal_patch[:, pos_idx] = ( gal_patch[:, pos_idx] / pixel_size - np.array([ix, iy, iz]) ) / npix_patch # normalize to [0,1)
        
        # Sort by sfr and pick top k = max_length
        if sort:
            sfr_patch = gal_patch[:, 0]  # the primary parameter
            sorted_indices = sorted(range(len(sfr_patch)), key=lambda k: sfr_patch[k], reverse=True)
            gal_patch = gal_patch[sorted_indices]
        gal_patch = gal_patch[:max_length]  # truncate to max_length

        # Append 
        x_list.append(torch.tensor(dm_density_patch, dtype=torch.float32))
        y_list.append(torch.tensor(gal_patch, dtype=torch.float32))
          
    return x_list, y_list

# ============================================================
# Mesh Dataset Base
# ============================================================

class MeshDatasetBase(Dataset):
    """
    Base class for Mesh Dataset. 
    """
    def __init__(
        self, 
        data_path,
        data_path_mesh,
        input_features,
        output_features,
        global_features = None,
        global_params = None,
        norm_param_dict = None,
        max_length = 100,
        npix_patch = 16,
        ndata = 1000,
        sort=True, 
        exclude_ratio=0,
        use_excluded_region=False,
        show_pbar=True
    ):
        
        if not isinstance(data_path, list):
            data_path = [data_path]
        if not isinstance(data_path_mesh, list):
            data_path_mesh = [data_path_mesh]

        if len(data_path) != len(data_path_mesh):
            raise ValueError("The number of paths and path_mesh must be the same.")
        
        self.x = []
        self.y = []
        self.g = []

        for i, (p, p_dm) in enumerate(zip(data_path, data_path_mesh)):

            dm_density, pixel_size = load_mesh_data(
                file_path = p_dm, 
                features = input_features, 
                norm_param_dict = norm_param_dict
                )
            gal_data, g_tmp = load_galaxy_data(
                file_path = p, 
                features = output_features, 
                global_features = global_features, 
                norm_param_dict = norm_param_dict
            )

            x_tmp, y_tmp = get_random_patches(
                dm_density, 
                gal_data, 
                ndata, 
                npix_patch, 
                pixel_size, 
                output_features, 
                max_length=max_length, 
                sort=sort, 
                exclude_ratio=exclude_ratio, 
                use_excluded_region=use_excluded_region, 
                show_pbar=show_pbar
                )
            
            self.x = self.x + x_tmp
            self.y = self.y + y_tmp

            if global_params is not None:       
                global_param = global_params[i]
                for ig, g in enumerate(g_tmp):
                    if g is not None:
                        global_param[ig] = g

                if np.isnan(global_param).any():
                    raise ValueError("global_params still contains Nan Values. Some missing global features may not have been replaced.")
                
                g_tmp = np.repeat(global_param[None, :], len(x_tmp), axis=0) # (Nhalo, num_features_global)

            else:
                g_tmp = np.zeros((len(x_tmp), 1)) # dummy (Nhalo, 1)

            self.g.append( g_tmp )

        self.x = torch.tensor( np.stack(self.x, axis=0), dtype=torch.float32) # (num_patches, npix_patch, npix_patch, npix_patch, num_features_cond)
        self.x = self.x.permute(0, 4, 1, 2, 3) # (num_patches, num_features_cond, npix_patch, npix_patch, npix_patch)  

        self.g = np.vstack(self.g)
        self.g = torch.tensor(self.g, dtype=torch.float32)

        self.pixel_size = pixel_size
        self.max_length = max_length
        self.pos_idx = get_index_list(output_features, "SubhaloPos")
        self.vel_idx = get_index_list(output_features, "SubhaloVel")

    def __len__(self):
        return len(self.x)
    
    def _axis_permutation(self, x, y):

        perm = torch.randperm(3)

        x = x.permute(int(perm[0]), int(perm[1]), int(perm[2]))

        y[..., self.pos_idx] = y[..., self.pos_idx].index_select(-1, perm)            # (...,3)
        if self.vel_idx[0] is not None:
            y[..., self.vel_idx] = y[..., self.vel_idx].index_select(-1, perm)           # (...,3)

        return x, y

    def _axis_flip(self, x, y):
        flips = (torch.rand(3) < 0.5)  # (3,) bool

        for ax in range(3):
            if bool(flips[ax]):
                x = torch.flip(x, dims=(ax,))

        sign = torch.where(flips, torch.tensor(-1.0, device=y.device), torch.tensor(1.0, device=y.device))

        y[..., self.pos_idx] = y[..., self.pos_idx] * sign
        if self.vel_idx[0] is not None:
            y[..., self.vel_idx] = y[..., self.vel_idx] * sign

        return x, y

    def __getitem__(self, idx):
        raise NotImplementedError("getitem is not implemented!")

# ============================================================
# Mesh Dataset
# ============================================================

class MeshDataset(MeshDatasetBase):
    """
    Dataset for loading mesh and galaxy data. 
    Galaxies are NOT used as context.
    """
    def __init__(
        self, 
        data_path,
        data_path_mesh,
        input_features,
        output_features,
        global_features = None,
        global_params = None,
        norm_param_dict = None,
        max_length = 100,
        npix_patch = 16,
        ndata = 1000,
        use_flat_representation = False,
        sort=True, 
        exclude_ratio=0, 
        use_excluded_region=False,
        show_pbar=True
    ):
        super().__init__(data_path = data_path, 
                         data_path_mesh = data_path_mesh,
                         input_features = input_features,
                         output_features = output_features,
                         global_features = global_features,
                         global_params = global_params, 
                         norm_param_dict = norm_param_dict,
                         max_length = max_length,
                         npix_patch = npix_patch,
                         ndata = ndata,
                         sort = sort, 
                         exclude_ratio = exclude_ratio, use_excluded_region = use_excluded_region, 
                         show_pbar = show_pbar
                         )

        _, num_params = (self.y[0]).shape
        self.y_padded = torch.zeros(len(self.x), max_length, num_params)
        self.mask = torch.zeros(len(self.x), max_length, num_params, dtype=torch.bool)
        
        for i, y_i in enumerate(self.y):
            length = len(y_i)
            self.y_padded[i, :length, :] = y_i[:max_length]
            self.mask[i, :length+1, :] = True # use the last + 1 value to learn when to stop

        self.use_flat_representation = use_flat_representation
                
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y_padded[idx].clone()
        mask = self.mask[idx]

        # Random rotation
        x, y = self._axis_permutation(x, y)

        # Random flip
        x, y = self._axis_flip(x, y)

        if self.use_flat_representation:
            y = y.reshape(-1, 1) # (max_length * output_features, 1)
            mask = mask.reshape(-1, 1) # (max_length * output_features, 1)

        out = {
            "condition": 
                {
                    "mesh3d": x
                },
            "target": y,
            "mask": mask,
            "global_cond": torch.zeros(1),
        }
        return out
    
# ============================================================
# Mesh + Ctx Dataset
# ============================================================
    
class MeshCtxDataset(MeshDatasetBase):
    """
    Dataset for loading mesh and galaxy data.
    Galaxies are used as context data in addition to target data.
    """
    def __init__(
        self, 
        data_path,
        data_path_mesh,
        input_features,
        output_features,
        global_features = None,
        global_params = None,
        norm_param_dict = None,
        max_length = 100,
        npix_patch = 16,
        ndata = 1000,
        use_flat_representation = False,
        sort = True, 
        exclude_ratio = 0, 
        use_excluded_region = False,
        round_id = None,
        show_pbar = True,
    ):
        super().__init__(data_path = data_path, 
                         data_path_mesh = data_path_mesh,
                         input_features = input_features,
                         output_features = output_features,
                         global_features = global_features,
                         global_params = global_params, 
                         norm_param_dict = norm_param_dict,
                         max_length = max_length,
                         npix_patch = npix_patch,
                         ndata = ndata,
                         sort = sort, 
                         exclude_ratio = exclude_ratio, use_excluded_region = use_excluded_region, 
                         show_pbar = show_pbar
                        )
        
        self.max_length = max_length
        self.use_flat_representation = use_flat_representation
        self.round_id = round_id

        _, num_params = self.y[0].shape
        self.y_tgt = torch.zeros(len(self.x), max_length, num_params)
        self.y_ctx = torch.zeros(len(self.x), max_length, num_params)
        self.mask_tgt = torch.zeros(len(self.x), max_length, num_params, dtype=torch.bool)
        self.mask_ctx = torch.zeros(len(self.x), max_length, num_params, dtype=torch.bool)
        self.boundary = torch.zeros(len(self.x), 6)

        for i, y in enumerate(self.y):
            check_fn, lo, hi = make_region_fn(round_id, self.pos_idx)
            self.boundary[i] = torch.cat([lo,hi])
            in_tgt_region = check_fn(y)

            y_tgt = y[in_tgt_region]
            self.y_tgt[i, :len(y_tgt)] = y_tgt
            self.mask_tgt[i, :len(y_tgt) + 1] = True

            y_ctx = y[~in_tgt_region]
            self.y_ctx[i, :len(y_ctx)] = y_ctx
            self.mask_ctx[i, :len(y_ctx)] = True

        if use_flat_representation:
            self.y_tgt = self.y_tgt.reshape(len(self.x), -1, 1)
            self.y_ctx = self.y_ctx.reshape(len(self.x), -1, 1)
            self.mask_tgt = self.mask_tgt.reshape(len(self.x), -1, 1)
            self.mask_ctx = self.mask_ctx.reshape(len(self.x), -1, 1)
            
    def __getitem__(self, idx):
        out = {
            "condition": 
                { 
                    "mesh3d": self.x[idx],
                    "context": self.y_ctx[idx],
                    "mask_ctx": self.mask_ctx[idx],
                    "boundary": self.boundary[idx]
                },
            "target": self.y_tgt[idx],
            "mask": self.mask_tgt[idx],
            "global_cond": torch.zeros(1),
        }

        return out
    
class MeshCtxAugmentedDataset(MeshDatasetBase):
    """
    Dataset for loading mesh and galaxy data.
    Galaxies are used as context data in addition to target data.
    
    Data structure is same as MeshCtxDataset, but here data qugmentation is applied.
    Note that the data augumentation is not optimized yet.
    """

    def __init__(
        self,
        data_path,
        data_path_mesh,
        input_features,
        output_features,
        global_features = None,
        global_params = None,
        norm_param_dict = None,
        max_length = 100,
        npix_patch = 16,
        ndata = 1000,
        use_flat_representation = False, 
        sort = True, 
        exclude_ratio = 0, 
        use_excluded_region = False,
        round_id = None,
        show_pbar = True,
    ):
        super().__init__(data_path = data_path, 
                         data_path_mesh = data_path_mesh,
                         input_features = input_features,
                         output_features = output_features,
                         global_features = global_features,
                         global_params = global_params, 
                         norm_param_dict = norm_param_dict,
                         max_length = max_length,
                         npix_patch = npix_patch,
                         ndata = ndata,
                         sort = sort, 
                         exclude_ratio = exclude_ratio, 
                         use_excluded_region = use_excluded_region, 
                         show_pbar = show_pbar
                         )

    def _pad_with_mask(self, y, buff=0):
        length = len(y)
        y_padded = torch.zeros(self.max_length, y.shape[-1])
        mask = torch.zeros(self.max_length, y.shape[-1], dtype=torch.bool)

        y_padded[:length] = y[:self.max_length]
        mask[:length+buff] = True

        return y_padded, mask
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        # Random rotation
        x, y = self._axis_permutation(x, y)

        # Random flip
        x, y = self._axis_flip(x, y)

        check_fn, lo, hi = make_region_fn(self.round_id, self.pos_idx)
        in_tgt_region = check_fn(y)

        y_tgt = y[in_tgt_region]
        y_ctx = y[~in_tgt_region]

        y_tgt_pad, mask_tgt = self._pad_with_mask(y_tgt, buff=1) # (max_length, num_params)
        y_ctx_pad, mask_ctx = self._pad_with_mask(y_ctx) # (max_length, num_params)

        if self.use_flat_representation:
            y_tgt_pad = y_tgt_pad.reshape(-1, 1)
            y_ctx_pad = y_ctx_pad.reshape(-1, 1)
            mask_tgt = mask_tgt.reshape(-1, 1)
            mask_ctx = mask_ctx.reshape(-1, 1)

        out = {
            "condition": 
                { 
                    "mesh3d": x,
                    "context": y_ctx_pad,
                    "mask_ctx": mask_ctx,
                    "boundary": torch.cat([lo, hi], dim=0)
                },
            "target": y_tgt_pad,
            "mask": mask_tgt,
            "global_cond": torch.zeros(1)
        }

        return out


# base regions in [0,1]
_LO = torch.tensor([
    [0.00, 0.00, 0.00],
    [0.25, 0.00, 0.00],
    [0.00, 0.25, 0.00],
    [0.25, 0.25, 0.00],
    [0.00, 0.00, 0.25],
    [0.25, 0.00, 0.25],
    [0.00, 0.25, 0.25],
    [0.25, 0.25, 0.25],
], dtype=torch.float32)

_HI = torch.tensor([
    [1.00, 1.00, 1.00],
    [0.75, 1.00, 1.00],
    [1.00, 0.75, 1.00],
    [0.75, 0.75, 1.00],
    [1.00, 1.00, 0.75],
    [0.75, 1.00, 0.75],
    [1.00, 0.75, 0.75],
    [0.75, 0.75, 0.75],
], dtype=torch.float32)

def round_offsets(round_id, npix_patch):
    """
    Input:
        round_id: int
        npix_patch: int
    Output:
        tuple
    """
    h = npix_patch // 2
    tbl = {
        0: (0, 0, 0),
        1: (h, 0, 0),
        2: (0, h, 0),
        3: (h, h, 0),
        4: (0, 0, h),
        5: (h, 0, h),
        6: (0, h, h),
        7: (h, h, h),
    }
    if round_id not in tbl:
        raise ValueError("Round id shoud be < 8.")
    
    return tbl[round_id]

def get_new_region(round_id, device='cpu'):
    """
    Input: 
        round_id: int
    Output:
        lo: torch.tensor
        hi: torch.tensor
    """
    lo = _LO[round_id].to(device)
    hi = _HI[round_id].to(device)
            
    return lo, hi

def make_region_fn(round_id=None, pos_idx=[0,1,2]):

    rid = random.randrange(8) if round_id is None else round_id
    lo, hi = get_new_region(rid)

    def check_fn(y):
        pos = y[..., pos_idx] 
        inside = ((pos >= lo) & (pos < hi)).all(dim=-1)
        return inside
    
    return check_fn, lo, hi
