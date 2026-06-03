import numpy as np
import h5py

#from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)


cspeed = 3e10  # [cm/s]
micron = 1e-4 # [cm]
GHz = 1e9
Jy = 1.0e-23        # jansky (erg/s/cm2/Hz)
arcsec = 4.848136811094e-6 # [rad] ... arcmin / 60 //

from line_model import calc_line_luminosity, line_dict
from cosmoglint.utils.cosmology_utils import z_to_log_lumi_dis

def create_line_intensity_map(
        pos_x, 
        pos_y, 
        z_obs, 
        z_real, 
        log_sfr,
        fmin,
        fmax,
        R,
        side_length,
        angular_resolution,
        line_list,
        intensity_unit = "Jy/sr",
        sigma = 0.2,
        args = None
    ):

    """
    Input:
        pos_x : np.ndarray, shape (N,) [arcsec]
        pos_y : np.ndarray, shape (N,) [arcsec]
        z_obs: np.ndarray, shape (N,)
            Observed redshift
        z_real: np.ndarray, shape (N,) 
            Real redshift
        log_sfr: np.ndarray, shape (N,)
        fmin: float [Hz]
        fmax: float [Hz]
        R: float
        side_length: float [arcsec]
        angular_resolution: float [arcsec]
        line_list: list of line names 
        intensity_unit: str, 
    """

    ### Initialize the data cube and flist
    flist = []
    dflist = []

    fnow = fmin * GHz
    while fnow <= fmax * GHz:
        flist.append(fnow)
        dflist.append(fnow / R)
        fnow += fnow / R
    flist = np.array(flist, dtype=np.float32)
    dflist = np.array(dflist, dtype=np.float32)

    Nx = int(side_length / angular_resolution)
    Nz = len(flist) - 1

    npix = np.array([Nx, Nx, Nz])
    total_intensity = np.zeros((Nx, Nx, Nz), dtype=np.float32)

    ix = np.floor(pos_x[:,0] / angular_resolution).astype(np.int32)
    iy = np.floor(pos_y[:,1] / angular_resolution).astype(np.int32)

    with h5py.File(args.output_fname, "w") as f:

        ### Save metadata
        if args is not None:
            args_dict = vars(args)
            args_dict = {k: (v if v is not None else "None") for k, v in args_dict.items()}
            for key, value in args_dict.items():
                f.attrs[key] = value

        f.create_dataset("frequency", data=flist, compression="gzip")

        ### Save intensities 
        for line_name in line_list:
            freq_obs = line_dict[line_name][0] / ( 1. + z_obs[:,2] )

            iz = np.searchsorted(flist, freq_obs, side="right") - 1

            indices = np.array([ix, iy, iz]).T # (num_galaxies, 3)

            valid_mask = np.all((indices >= 0) & (indices < npix), axis=1)
            
            intensity_line = np.zeros((Nx, Nx, Nz), dtype=np.float32)

            if np.sum(valid_mask) > 0:
                indices_valid = indices[valid_mask]
                z_valid = z_real[valid_mask]
                log_sfr_valid = log_sfr[valid_mask]

                log_lumi = calc_line_luminosity(args, z_valid, log_sfr_valid, line_name, sigma=sigma)
                log_lumi_dis = z_to_log_lumi_dis(z_valid, cosmo) # [cm]

                flux = 10 ** ( log_lumi - 2 * log_lumi_dis ) / ( 4. * np.pi ) # [erg/s/cm2]

                print("# Found {} valid galaxies for {}; Total flux {:.3e}".format(np.sum(valid_mask), line_name, np.sum(flux)))

                if intensity_unit == "erg/s/cm2/Hz/beam":
                    intensity_valid = flux / dflist[indices_valid[:, 2]] # [erg/s/cm2/Hz]
                elif intensity_unit == "erg/s/cm2/sr":
                    intensity_valid = flux / ( angular_resolution * arcsec )**2
                else: # Default: "Jy/sr"
                    intensity_valid = flux / dflist[indices_valid[:, 2]] / Jy / ( angular_resolution * arcsec )**2 # [Jy/sr]

                np.add.at(intensity_line, (indices_valid[:, 0], indices_valid[:, 1], indices_valid[:, 2]), intensity_valid)

                total_intensity += intensity_line

                f.create_dataset("intensity_{}".format(line_name), data=intensity_line, compression="gzip")

        f.create_dataset("total_intensity", data=total_intensity, compression="gzip")

    print("Intensity map saved to {}".format(args.output_fname))
