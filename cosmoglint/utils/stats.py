### Functions for computing anuglar power spectrum
### based on power_box's implementation

import sys
try:
    from multiprocessing import cpu_count
    THREADS = cpu_count()
    from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, fftfreq
    def fftn(*args, **kwargs):
        return _fftn(threads=THREADS, *args, **kwargs)
    def ifftn(*args, **kwargs):
        return _ifftn(threads=THREADS, *args, **kwargs)
except ImportError:
    print("Warning: you are not using FFTW.", file=sys.stderr)
    from numpy.fft import fftn, ifftn, fftfreq 
import numpy as np # import of numpy should be done after pyfftw import
import warnings, os

def short_formatwarning(msg, category, filename, lineno, line=None):
    return f"{os.path.basename(filename)}:{lineno}: {category.__name__}: {msg}\n"

warnings.formatwarning = short_formatwarning
warnings.filterwarnings("always", category=RuntimeWarning)

def my_fft(X, L=None, b=1.): # b = 2 * np.pi for inverse FT

    dim = len(X.shape)
    N = np.array(X.shape)

    if L is None:
        L = N
    elif np.isscalar(L):
        L = L*np.ones(dim)

    dx = np.array([float(l)/float(n) for l, n in zip(L,N)]) # array of side length of cell
    Vx = np.prod(dx) # volume of cell

    ### compute fft ###
    freq = [fftfreq(n, d=d)*2*np.pi for n, d in zip(N, dx)] ## k = 2pi * f
    ft = Vx * fftn(X) * np.sqrt(1/np.abs(b)) ** dim

    return ft, freq

def angular_average_nd(field, coords, nbins, n=None, log_bins=False, indx_min=0, verbose=True):
    ## can be used for real field only

    dim = len(field.shape)
    if len(coords) != dim:
        raise ValueError("coords should be a list of arrays, one for each dimensiont.")

    coords = np.sqrt(np.sum(np.meshgrid(*([x**2] for x in coords)), axis=0)) # k(i,j) = sqrt( ki*ki + kj*kj )

    ### define bins ###
    # "bins" here includes nbins + 1 components corresponding to the edge values of the k-bin
    # if you want to compute the "k of the bin", then you should take (bins[i]+bins[i+1])/2
    mx = coords.max()
    if not log_bins:
        bins = np.linspace(coords.min(), mx, nbins + 1)
    else:
        mn = coords[coords>0].min()
        bins = np.logspace(np.log10(mn), np.log10(mx), nbins + 1)

    ### set index for each pixel ###
    # label the field with a indx that satisfies bins[indx] <= coords < bins[indx+1]
    # indx takes 0, 1, ..., nbin+1
    indx = np.digitize(coords.flatten(), bins)

    ### compute the number of pixels in each bin ###
    # each component in bincount correaponds to the number of pixels with indx = 0, 1, ..., nbin+1
    # i.e., bincount has nbin + 2 = len(bins) + 1 components
    # the first component of bincount is for values < mn.
    # the last component of bincount is for values > mx, where there should be no such pixels. So bincount[-1] is always 0.
    # we remove these two compontnes by setting [1:-1].
    sumweights = np.bincount(indx, minlength=len(bins)+1)[1:-1] 
    if verbose:
        if np.any(sumweights==0):
            print("Warning: one or more radial bins had no cell within it. Use a smaller nbins.")
        if np.any(sumweights==1):
            print("Warning: one or more radial bins have only one cell within it. This would result in inf in the variance")

    ### compute the mean in each bin ###
    # for each bin, sum up the field values, and then divide by the number of pixels
    mean = np.bincount(indx, weights=np.real(field.flatten()), minlength=len(bins)+1)[1:-1] / sumweights

    ### compute variance ### 
     # for each bin, sum up the variance field values (i.e., (field-average)**2), and then divide by the number of pixels
    average_field = np.concatenate(([0], mean, [0]))[indx]
    var_field = ( field.flatten() - average_field ) ** 2
    var = np.bincount(indx, weights=var_field, minlength=len(bins)+1)[1:-1] / (sumweights-1) 

    return mean[indx_min:], bins[indx_min:], var[indx_min:]

def compute_power(deltax, deltax2=None, boxlength=1., nbins=20, log_bins=False, verbose=True):
    ## compmute power spectrum 
    # Inputs:
    #   deltax: input image 
    #   deltax2: second input image. If this is not None, the cross-power spectrum will be computed
    #   boxlength: float or list of floats. The physical length of the side(s) in real space
    #   nbins: int. number of bins
    #   log_bins: use log-scale bins if True.
    #
    # Outputs: 
    #   Pk: angular-averaged power spectrum. 
    #   k: edge values of the bins. Note there are N+1 values when the length of Pk is N.
    #   var: variance 

    if deltax2 is not None and deltax.shape != deltax2.shape:
        raise ValueError("deltax and deltax2 must have the same shpae!")

    dim = len(deltax.shape)

    if not np.iterable(boxlength):
        boxlength = [boxlength] * dim
    V = np.prod(boxlength)

    # Fourier transformation
    FT, freq = my_fft(deltax, L=boxlength) 

    if deltax2 is not None:
        FT2, _ = my_fft(deltax2, L=boxlength)
    else:
        FT2 = FT

    # compute power spectrum
    P = np.real(FT * np.conj(FT2)) / V
    """ we need to implement the window function...
    wx = np.meshgrid(*([np.sin(k*l/2.)/(k*l/2.)] for k, l in zip(freq, boxlength))) ## array (dim, N, ..., N); each component ks[i] is the wavenumber map of i-th dimension
    window = np.ones(wx[0].shape)
    for w in wx:
        window = np.multiply(window, w)
    P = P / window**2
    """ 

    # compute angular power spectrum
    Pk, k, var = angular_average_nd(P, freq, nbins, log_bins=log_bins, verbose=verbose)

    return Pk, k, var

def cylindrical_average(field, coords, nbins, log_bins=False, use_same_bins=False, verbose=True):

    if field.ndim != 3 or len(coords) != 3:
        raise ValueError("field must be 3D and coords must be a list of 3 1D arrays.")

    X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')

    r_coords = np.sqrt(X**2 + Y**2)
    z_coords = np.abs(Z)

    # binning
    z_min, z_max = z_coords.min(), z_coords.max()
    r_min, r_max = r_coords.min(), r_coords.max() 
    if log_bins:
        z_min = z_coords[z_coords > 0].min()
        r_min = r_coords[r_coords > 0].min()   
    if use_same_bins:
        z_min = min(z_min, r_min)
        z_max = max(z_max, r_max)
        r_min = z_min
        r_max = z_max
        
    if not log_bins:
        bins_z = np.linspace(z_min, z_max, nbins + 1)
    else:
        bins_z = np.logspace(np.log10(z_min), np.log10(z_max), nbins + 1)
    
    if not log_bins:
        bins_r = np.linspace(r_min, r_max, nbins + 1)
    else:
        bins_r = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)

    indx_r = np.digitize(r_coords.flatten(), bins_r)
    indx_z = np.digitize(z_coords.flatten(), bins_z)

    valid = (indx_r >= 1) & (indx_r <= nbins) & (indx_z >= 1) & (indx_z <= nbins)
    indx_r = indx_r[valid] - 1  
    indx_z = indx_z[valid] - 1

    combined_idx = indx_r + nbins * indx_z

    sumweights = np.bincount(combined_idx, minlength=nbins * nbins)
    sumweights = sumweights.reshape((nbins, nbins))
    if verbose:
        if np.any(sumweights == 0):
            print("Warning: one or more radial bins had no cell within it. Use a smaller nbins.")
        if np.any(sumweights == 1):
            print("Warning: one or more radial bins have only one cell within it. This would result in inf in the variance")

    field_flat = field.flatten()
    sum_field = np.bincount(combined_idx, weights=field_flat[valid], minlength=nbins * nbins)
    sum_field = sum_field.reshape((nbins, nbins))
    mean = sum_field / sumweights

    average_arr = mean.flatten()  # shape (nbins*nbins,)
    averaged_field = average_arr[combined_idx] 
    var_field = (field_flat[valid] - averaged_field) ** 2
    var = np.bincount(combined_idx, weights=var_field, minlength=nbins * nbins)
    var = var.reshape((nbins, nbins))
    var = var / (sumweights - 1)  
     
    return mean, bins_r, bins_z, var


def compute_cylindrical_power(deltax, deltax2=None, boxlength=1., nbins=10, log_bins=False, use_same_bins=False):
    ## compmute 2d power spectrum (kpara, kperp)
    # Inputs:
    #   deltax: input image 
    #   deltax2: second input image. If this is not None, the cross-power spectrum will be computed
    #   boxlength: float or list of floats. The physical length of the side(s) in real space
    #   nbins: int. number of bins
    #   log_bins: use log-scale bins if True.
    #
    # Outputs: 
    #   Pk: angular-averaged power spectrum. 
    #   k: edge values of the bins. Note there are N+1 values when the length of Pk is N.
    #   var: variance 

    if deltax2 is not None and deltax.shape != deltax2.shape:
        raise ValueError("deltax and deltax2 must have the same shpae!")

    dim = len(deltax.shape)

    if not np.iterable(boxlength):
        boxlength = [boxlength] * dim
    V = np.prod(boxlength)

    # Fourier transformation
    FT, freq = my_fft(deltax, L=boxlength) 

    if deltax2 is not None:
        FT2, _ = my_fft(deltax2, L=boxlength)
    else:
        FT2 = FT

    # compute power spectrum
    P = np.real(FT * np.conj(FT2)) / V

    # compute angular power spectrum
    Pk, k_perp, k_para, var = cylindrical_average(P, freq, nbins, log_bins=log_bins, use_same_bins=use_same_bins)

    return Pk, k_perp, k_para, var

def normalized_power(deltax, deltax2=None, boxlength=1., nbins=20, log_bins=True):
    # compute normalized power spectrum
    # Pk / k^3
    #print(deltax.shape, boxlength)
    Pk, k, var = compute_power(deltax, deltax2=deltax2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    err = np.sqrt(var)

    if log_bins:
        k_values = 10**((np.log10(k[1:]) + np.log10(k[:-1])) * 0.5)
    else:
        k_values = (k[1:] + k[:-1]) * 0.5

    dim = len(deltax.shape)
    if dim == 1:
        norm_Pk = Pk * k_values
        norm_err = err * k_values
    elif dim == 2:
        norm_Pk = Pk * k_values**2 / (2*np.pi)
        norm_err = err * k_values**2 / (2*np.pi)
    elif dim == 3:
        norm_Pk = Pk * k_values**3 / (2*np.pi**2)
        norm_err = err * k_values**3 / (2*np.pi**2)
    else:
        print("Error: dim > 3 is not supported", file=sys.stderr)
        sys.exit(1)

    return norm_Pk, k, norm_err**2

def compute_r(image1, image2, boxlength=1., nbins=20, log_bins=True):
    # compute cross-correlation coefficient between image1 and image2: r = Px / sqrt( P1 * P2 )
    # boxlength: float or list of floats. The physical length of the side(s) in real space
    # nbins: int. number of bins
    # log_bins: use log-scale bins if True.
    Px, k, varx = compute_power(image1, deltax2=image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P1, _, var1 = compute_power(image1, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    P2, _, var2 = compute_power(image2, boxlength=boxlength, nbins=nbins, log_bins=log_bins)
    return Px / np.sqrt( P1 * P2 ), k

def calc_lightcone_noise_power(sigma_noise_Jy_sr, freq_obs, freq_rest, intensity, side_length=3600, line_name="CO(1-0)", with_hlittle=True, cosmo=None):

    from .cosmology_utils import arcsec_to_cMpc, freq_to_comdis

    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo_default = FlatLambdaCDM(H0=67.74, Om0=0.3089)
       
    redshifts = freq_rest / freq_obs - 1
    redshift_mean = np.mean(redshifts)

    Lx = arcsec_to_cMpc(side_length, redshift_mean, cosmo=cosmo, l_with_hlittle=with_hlittle) # [cMpc/h]
    Ly = arcsec_to_cMpc(side_length, redshift_mean, cosmo=cosmo, l_with_hlittle=with_hlittle) # [cMpc/h]
    Lz = freq_to_comdis(freq_obs[0], freq_rest, cosmo=cosmo, l_with_hlittle=with_hlittle) - freq_to_comdis(freq_obs[-1], freq_rest, cosmo=cosmo, l_with_hlittle=with_hlittle) 

    dx = Lx / intensity.shape[0]
    dy = Ly / intensity.shape[1]
    dz = Lz / intensity.shape[2]
    
    dL = np.array([dx, dy, dz])
    dV = np.prod(dL) 

    P_noise = sigma_noise_Jy_sr**2 * dV

    return P_noise


def calc_lightcone_power(freq_obs, freq_rest, intensity, intensity2=None, side_length=3600, line_name="CO(1-0)", dlogk=0.2, with_hlittle=True, logkpara_min=-10, logkperp_min=-10, sigma_noise=0, sigma_noise2=0, cosmo=None): 
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
    from .cosmology_utils import arcsec_to_cMpc, freq_to_comdis
    
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo_default = FlatLambdaCDM(H0=67.74, Om0=0.3089)

    redshifts = freq_rest / freq_obs - 1
    redshift_mean = np.mean(redshifts)
    
    print("Use {} rest-frame frequency".format(line_name))
    print("redshift: {:.2f} - {:.2f} (mean: {:.2f})".format(redshifts[-1],redshifts[0],redshift_mean))
    
    Lx = arcsec_to_cMpc(side_length, redshift_mean, cosmo=cosmo, l_with_hlittle=with_hlittle) # [cMpc]
    Ly = arcsec_to_cMpc(side_length, redshift_mean, cosmo=cosmo, l_with_hlittle=with_hlittle) # [cMpc]
    Lz = freq_to_comdis(freq_obs[0], freq_rest) - freq_to_comdis(freq_obs[-1], freq_rest) # [cMpc]

    ## Fourier transform
    L = np.array([Lx, Ly, Lz])
    dL = np.array([Lx / intensity.shape[0], Ly / intensity.shape[1], Lz / intensity.shape[2]])

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


import treecorr
def calc_corr(pos, boxsize, min_sep=0.3, max_sep=200, nbins=20, n_random=None, fout=None):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]
    if n_random is None:
        n_random = len(x)
    x_rand = np.random.uniform(0, boxsize, n_random)
    y_rand = np.random.uniform(0, boxsize, n_random)
    z_rand = np.random.uniform(0, boxsize, n_random)

    # ---- Create Catalog ----
    cat = treecorr.Catalog(x=x, y=y, z=z)
    cat_random = treecorr.Catalog(x=x_rand, y=y_rand, z=z_rand)

    # ---- Setup ----
    nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins) 
    dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins) 
    rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins) 

    # ---- Calculate correlation function ----
    nn.process(cat)
    dr.process(cat, cat_random)
    rr.process(cat_random, cat_random)

    if fout is not None:
        nn.write(fout, rr=rr, dr=dr)
        print("output: {}".format(fout))

    xi, varxi = nn.calculateXi(rr=rr, dr=dr)
    r = np.exp(nn.meanlogr)   

    return r, xi, varxi


def precompute_shell_offsets(r_min, r_max, nbins, voxel_size=1.0):

    shell_offsets = [[] for _ in range(nbins)]

    max_index_offset = int(np.ceil(r_max / voxel_size))

    #edges = np.linspace(r_min, r_max, nbins+1)
    edges = np.logspace(np.log10(r_min), np.log10(r_max), nbins+1)

    for dx in range(-max_index_offset, max_index_offset + 1):
        for dy in range(-max_index_offset, max_index_offset + 1):
            for dz in range(-max_index_offset, max_index_offset + 1):
                r = np.sqrt(dx*dx + dy*dy + dz*dz) * voxel_size
                
                if r < r_min or r >= r_max:
                    continue
                bin_idx = np.searchsorted(edges, r, side="right") - 1
                
                if bin_idx < 0 or bin_idx >= nbins:
                    continue

                shell_offsets[bin_idx].append((dx, dy, dz))

    shell_offsets = [np.array(o, dtype=np.int32) if len(o) > 0 else np.zeros((0, 3), dtype=np.int32)
                    for o in shell_offsets]
    
    #radii = (np.arange(nbins) + 0.5) * dr  
    radii = np.sqrt(edges[:-1] * edges[1:])
    return shell_offsets, radii


def radial_profile_around_points(mesh, points, r_min, r_max, nbins, voxel_size=1.0, origin=(0.0, 0.0, 0.0)):
    """
    mesh: (N, N, N)
    points: (L, 3) 
    r_max, dr: 
    voxel_size: 
    origin: position of mesh[0,0,0] 
    """
    N = mesh.shape[0]
    assert mesh.shape[1] == N and mesh.shape[2] == N, "mesh shape should be (N,N,N)"

    shell_offsets, radii = precompute_shell_offsets(r_min, r_max, nbins, voxel_size)

    nbins = len(shell_offsets)
    L = points.shape[0]

    profiles = np.full((L, nbins), np.nan, dtype=np.float64)

    idx_f = (points - np.array(origin)) / voxel_size
    idx = np.rint(idx_f).astype(int)  # (L, 3)

    for i in range(L):
        ix, iy, iz = idx[i]
        if not (0 <= ix < N and 0 <= iy < N and 0 <= iz < N):
            continue

        for b in range(nbins):
            offsets = shell_offsets[b]
            if offsets.shape[0] == 0:
                continue
            
            ox = ix + offsets[:, 0]
            oy = iy + offsets[:, 1]
            oz = iz + offsets[:, 2]

            mask = (ox >= 0) & (ox < N) & (oy >= 0) & (oy < N) & (oz >= 0) & (oz < N)
            if not np.any(mask):
                continue

            vals = mesh[ox[mask], oy[mask], oz[mask]]
            profiles[i, b] = vals.mean()

    return profiles, radii