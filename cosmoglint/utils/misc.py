
from argparse import Namespace
import numpy as np

def safe_index(lst, key):
    try: 
        return lst.index(key)
    except ValueError:
        return None
  
def get_index_list(features, key_prefix, n=3):
    idx_lst = [ safe_index(features, f"{key_prefix}:{i}") for i in range(n)]
    return idx_lst

def namespace_to_dict(ns):
    if isinstance(ns, Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    else:
        return ns


def make_density_map(pos, npix, weight=1.0, mode="CIC", periodic=False):

    map = np.zeros((npix,npix,npix), dtype=np.float64)

    if mode == "NGP":
        ix = pos.astype(int)
        ix = np.clip(ix, 0, npix - 1)
        map[ix[:,0], ix[:,1], ix[:,2]] += weight

    elif mode == "CIC":
        map_flat = map.reshape(-1)
        i0 = np.floor(pos).astype(np.int64)
        d = pos - i0
        i1 = i0 + 1
        if periodic:
            i1 = i1 % npix
        else:
            i1 = np.clip(i1, 0, npix-1)

        w0 = 1.0 - d
        w1 = d

        ix0, iy0, iz0 = i0[:,0], i0[:,1], i0[:,2]
        ix1, iy1, iz1 = i1[:,0], i1[:,1], i1[:,2]
        wx0, wy0, wz0 = w0[:,0], w0[:,1], w0[:,2]
        wx1, wy1, wz1 = w1[:,0], w1[:,1], w1[:,2]
        def add_triple(ix, iy, iz, w):
            idx = (ix * npix + iy) * npix + iz
            np.add.at(map_flat, idx, w)

        add_triple(ix0, iy0, iz0, wx0*wy0*wz0*weight)
        add_triple(ix1, iy0, iz0, wx1*wy0*wz0*weight)
        add_triple(ix0, iy1, iz0, wx0*wy1*wz0*weight)
        add_triple(ix1, iy1, iz0, wx1*wy1*wz0*weight)
        add_triple(ix0, iy0, iz1, wx0*wy0*wz1*weight)
        add_triple(ix1, iy0, iz1, wx1*wy0*wz1*weight)
        add_triple(ix0, iy1, iz1, wx0*wy1*wz1*weight)
        add_triple(ix1, iy1, iz1, wx1*wy1*wz1*weight)

        map = map_flat.reshape((npix, npix, npix))

    else:
        raise ValueError("Unknown mode: {}".format(mode))
    
    return map


def get_sampler(x, xmin, xmax, nbins=20, temperature=1, weight_min=1e-8):
        x = x.detach().to("cpu")
        bins = torch.linspace(xmin, xmax, steps=nbins+1)
        bin_indices = torch.bucketize(x, bins, right=False) - 1
        bin_indices = bin_indices.clamp(0, nbins-1)
        counts = torch.bincount(bin_indices, minlength=nbins).to(torch.double)
        weights = 1. / counts[bin_indices] 
        weights = weights.pow(temperature) # Apply temperature scaling
        weights = weights.clamp(min=weight_min) # Avoid zero weights
        # When setting replacement to True and num_samples to the original number of samples, the sampler can select the same sample multiple times even within a single epoch.
        # The minimum weight is set to balance the sampling (few samples appear less frequently than when minimum is not set) 
        # Large minimum weight (larger than ~1e-5: the maximum number of halo mass function at z = 2) means the rare samples will be sampled more frequently (could suffer from overfitting, but might be faster to converge)
        return WeightedRandomSampler(weights, len(weights), replacement=True)