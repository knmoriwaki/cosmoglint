# LIM Mock Generator

A Python package of generative models for line intensity mapping (LIM) experiments. 

---

## Installation

You can install the package directly from GitHub:

```
pip install git+https://github.com/knmoriwaki/lim-mock-generator.git
```

Or, if you cloned the repository locally:

```
git clone https://github.com/knmoriwaki/lim-mock-generator.git
cd lim-mock-generator
pip install -e .
```

## Training 

Put training data at `[data_path]` and run the following.
```
cd ./scripts
python train_transformer.py --data_path [data_path] --use_dist --use_vel
```
The distance and velocity relative to halo are modeled when options `--use_dist` and `--use_vel` are given.

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

## Create mock data

Put a halo catalog at `[input_fname]` and the trained model at `[model_dir]` and run the following.
```
cd ./scripts
python create_data.py --input_fname [input_fname] --model_dir [model_dir]
```

The halo catalog should be a text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns).

Currently, the package is designed to work with comoving volume box data only. Support for lightcone data may be added in a future update.

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `plot_transformer.ipynb`: visualize training results
- `plot_mock.ipynb`: visualize created mock data


## Pre-trained model

The pre-trained model trained with TNG300-1 data at z = 2 and an example input are available at [Google Drive](https://drive.google.com/drive/folders/1HRkRdfti8XaIPyF3er5QJmFX3WXCmAQI?usp=sharing).

## Citation
