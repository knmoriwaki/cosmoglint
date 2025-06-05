import numpy as np

GHz = 1e9
micron = 1e-4 # [cm]
angstrom = 1e-8 # [cm]

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
    #"[OIII]88": [3393.006 * GHz, 88.0 * micron],
    #"[NII]205": [1461.131 * GHz, 205.0 * micron],
    #"[NII]122": [2459.381 * GHz, 122.0 * micron],
    "[CI](1-0)": [492.16065 * GHz, 609.14 * micron],
    "[CI](2-1)": [809.34197 * GHz, 370.42 * micron],
    #"[OIII]5007": [5.997e5 * GHz, 5007.0 * angstrom],
}

FIR_SFR =  2.22e43
Lsun = 3.828e33  # erg/s

def calc_line_luminosity(args, z, log_sfr, line_name):
    
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
        
    if "CO" in line_name:
        # Convert [K km s-1 pc2] -> [erg/s] 
        # see eq. 28 of Fonseca+2017 or Carilli+2013 for this conversion.
        freq_rest = line_dict[line_name][0] # [Hz]
        log_lumi += np.log10( 1.227e-4 ) + 3.0 * np.log10( freq_rest ) 

    ### Add scatter
    r1 = np.random.rand(len(log_sfr))
    r2 = np.random.rand(len(log_sfr))
    log_lumi += args.sigma * np.sqrt( -2.0 * np.log(r1) ) * np.sin( 2.0 * np.pi * r2 )

    return log_lumi
