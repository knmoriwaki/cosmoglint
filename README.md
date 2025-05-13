# Generative model for LIM mocks

## Installation

Install the required libraries
```
pip install -r requirements.txt
```

The following software packages are additionally needed for running Pinocchio 
- gsl (https://www.gnu.org/software/gsl/)
- cfitso (https://heasarc.gsfc.nasa.gov/fitsio/fitsio_macosx.html)
- fftw (http://www.fftw.org)


## Training 

Put training data at `[data_path]` and run the following.
```
cd Transformer
python3 main.py --data_path [data_path] --use_dist --use_vel
```
The distance and velocity relative to halo are modeled when options `use_dist` and `use_vel` are given.

The data should be a hdf5 file that contains the following properties of halos:
- `HaloMass` 
- `NumSubgroups` 
- `Offset` 

and the following properties of galaxies:
- `SubgroupSFR` 
- `SubgroupDist` 
- `SubgroupVrad` 
- `SubgroupVtan` 

`NumSubgroups` and `Offset` are used for determining host halos of galaxies.

## Check results 

Use `Transformer/plot.ipynb`.

## Create mock data

Put a halo catalog at `[input_fname]` and the trained model at `[model_dir]` and run the following.
```
python3 create_data_cube.py --input_fname [input_fname] --model_dir [model_dir]
```

The halo catalog should be a text file that contains comving positions at 1st to 3rd columns and velocities at 4th to 7th columns.

## Pre-trained model

The pre-trained model trained with TNG300-1 data at z = 2 is available at [Google Drive](https://drive.google.com/drive/folders/1HRkRdfti8XaIPyF3er5QJmFX3WXCmAQI?usp=sharing).

## Citation

