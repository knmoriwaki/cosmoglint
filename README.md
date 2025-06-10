# Transformer-based Mock Generator for Line Intensity Mapping 

Model to generate galaxy properties for a given halo mass.

A model trained with TNG300-1 at z = 0.5 - 6 as well as example input data for training and mock creation are available at [Google Drive](https://drive.google.com/drive/folders/1IFje9tNRf4Dr3NufqzlDdGMFTEDpsm35?usp=share_link).

Currently, the package is designed to work with comoving volume box data only. Support for lightcone data may be added in a future update.

For detailed usage and options, see [DOCUMENTATION](./DOCUMENTATION.md).

---

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/knmoriwaki/lim-mock-generator.git
```

Install from local clone:

```bash
git clone https://github.com/knmoriwaki/lim-mock-generator.git
cd lim-mock-generator
pip install .
```

Several libraries needs to be additionally installed to run the scripts and notebooks:
```bash
pip install -r requirements.txt
```

## Training 

Example:
```bash
python train_transformer.py --data_path [data_path] --norm_param_file [norm_param_file] --use_dist --use_vel
```

Options:
- `--data_path`: Path(s) to the training data. Data is an hdf5 file that contains properties of halos (`HaloMass`, `NumSubgroups`, `Offset`) and galaxies (`SubgroupSFR`, `SubgroupDist`, `SubgroupVrad`, `SubgroupVtan`), where `NumSubgroups` and `Offset` are used for determining host halos of galaxies. Multiple files can be passed.
- `--norm_param_file`: Path to the normalization parameter file. The file contains minimum (1st column) and maximum (2nd column) values for the input and output parameters.
- `--use_dist`: If set, the distance to halo is predicted
- `--use_vel`: If set, relative velocity to halo is predicted

## Create mock data

Example:
```bash
python create_data_cube.py --input_fname [input_fname] --model_dir [model_dir] 
```

Options:
- `--input_fname`: Path to the halo catalog. Text file that contains the logarithmic total halo mass [Msun] (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns) and catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format are supported.
- `--model_dir`: Path to a directory containing the trained model (`model.pth` and `args.json`). If not set, column 7 of the input file is used as intensity.

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `plot_transformer.ipynb`: visualize training results
- `plot_data_cube.ipynb`: visualize created data cube


## Citation

