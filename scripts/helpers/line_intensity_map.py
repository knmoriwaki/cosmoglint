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
angstrom = 1e-8 # [cm]

from cosmoglint.utils.cosmology_utils import z_to_log_lumi_dis

def create_and_save_line_intensity_map(
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
    "Ha": [4.568e5 * GHz, 6562.8 * angstrom],  
    "[OIII]5007": [5.997e5 * GHz, 5007.0 * angstrom],
}


FIR_SFR =  2.22e43
Lsun = 3.828e33  # erg/s

def calc_line_luminosity(z, log_sfr, line_name, sigma=0.2):
    
    #ssfr = 10 ** (log_sfr - log_mstar)
    logL_FIR = log_sfr + np.log10( FIR_SFR / Lsun ) # [Lsun]: See eq. 23 of Fonseca+2017

    if line_name == "CO(1-0)" or line_name == "CO(2-1)" or line_name == "CO(3-2)":
        log_lumi = 0.81 * logL_FIR + 0.54 # Sargent+ 14 [K km s-1 pc2]
        #log_lumi[ssfr > 0.5e-8] -= 0.46 # starbursting galaxies. Criteria is from Fig. 1 of Sargent+14
    elif line_name == "CO(2-1)":
        log_lumi += 0.76 # CO ratio from Daddi+15 [K km s-1 pc2]
    elif line_name == "CO(3-2)":
        #log_lumi = log_sfr + np.log10( 1.0e8 ) # Papadopoulos 12
        #log_lumi = log_sfr + np.log10( 3.2e8 ) # Popping+ 18 average [K km s-1 pc2]
        log_lumi += 0.42 # CO ratio from Daddi+15 [K km s-1 pc2]
    elif line_name == "CO(4-3)": 
        log_lumi = ( logL_FIR - 1.49 ) / 1.06 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(5-4)":
        log_lumi = ( logL_FIR - 1.71 ) / 1.07 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(6-5)":
        log_lumi = ( logL_FIR - 1.79 ) / 1.10 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(7-6)":
        log_lumi = ( logL_FIR - 2.62 ) / 1.03 # Liu+15 [K km s-1 pc2]	
    elif line_name == "CO(8-7)":
        log_lumi = ( logL_FIR - 2.82 ) / 1.02 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(9-8)":
        log_lumi = ( logL_FIR - 3.10 ) / 1.01 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(10-9)":
        log_lumi = ( logL_FIR - 3.67 ) / 0.96 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(11-10)":
        log_lumi = ( logL_FIR - 3.51 ) / 1.00 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(12-11)":
        log_lumi = ( logL_FIR - 3.83 ) / 0.99 # Liu+15 [K km s-1 pc2]
    elif line_name == "CO(13-12)":
        log_lumi = logL_FIR - 5.
    elif line_name == "[CII]158":
        log_lumi = ( 6.99 + log_sfr ) / 1.01 + np.log10( Lsun ) #DeLooze+14 [erg/s]
    elif line_name == "[OIII]88":
        log_lumi = ( 7.48 + log_sfr ) / 1.12 + np.log10( Lsun ) #DeLooze+14 [erg/s]
    elif line_name == "[NII]205":
        log_lumi = log_sfr + np.log10( 2.5 * 1.0e5 * Lsun ) # Visbal & Loeb [erg/s]
    elif line_name == "[NII]122":
        log_lumi = log_sfr + np.log10( 7.9 * 1.0e5 * Lsun ) # Visbal & Loeb [erg/s]
    elif line_name == "[CI](1-0)" or line_name == "[CI](2-1)":
        log_lumi = ( logL_FIR - 1.49 ) / 1.06 + np.log10( 1.227e-4 ) + 3.0 * np.log10( line_dict["CO(4-3)"][0] ) # CO(4-3)
        log_lumi = 1.07 * ( log_lumi - np.log10( Lsun ) - logL_FIR ) + 0.14 + logL_FIR + np.log10( Lsun ) # Bethermin+22 Eq. 9 [erg/s]
    elif line_name == "[CI](2-1)":
        diff = ( logL_FIR - 2.62 ) / 1.03 - ( logL_FIR - 1.49 ) / 1.06 # CO(7-6) - CO(4-3)
        log_lumi = log_lumi + 0.63 * ( diff ) + 0.17 # Bethermin+22 Eq. 10 [erg/s]
    elif line_name == "[OIII]5007":
        log_lumi = log_sfr + np.log10( 1.32e41 ) # [erg/s]
    elif line_name == "Ha":
        log_lumi = log_sfr + np.log10( 1.2e41 ) # [erg/s] Kennicutt+1998 Eq. 2 (Salpeter IMF)

    if "CO" in line_name:
        # Convert [K km s-1 pc2] -> [erg/s] 
        # see eq. 28 of Fonseca+2017 or Carilli+2013 for this conversion.
        freq_rest = line_dict[line_name][0] # [Hz]
        log_lumi += np.log10( 1.227e-4 ) + 3.0 * np.log10( freq_rest ) 

    ### Add scatter
    r1 = np.random.rand(len(log_sfr))
    r2 = np.random.rand(len(log_sfr))
    log_lumi += sigma * np.sqrt( -2.0 * np.log(r1) ) * np.sin( 2.0 * np.pi * r2 )

    return log_lumi
