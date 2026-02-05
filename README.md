# CosmoGLINT: Cosmological Generative model for Line INtensity mapping with Transformer

This repository includes:

- cosmoglint, a package of Transformer-based models that generate galaxy properties from halo mass.
- Scripts for training and mock catalog generation.
- Example notebooks for result visualization.

Models trained with TNG300-1 at z = 0.5 - 6 and generated data are available at [Google Drive](https://drive.google.com/drive/folders/1IFje9tNRf4Dr3NufqzlDdGMFTEDpsm35?usp=share_link).

For detailed usage and options, see [DOCUMENTATION](./DOCUMENTATION.md).

---

## Installation

Python>=3.9 is required. 

This package requires PyTorch.
Please install PyTorch first following https://pytorch.org

Install package and from local clone:

```bash
git clone https://github.com/knmoriwaki/cosmoglint.git
cd cosmoglint
pip install .
```

If you only need the `cosmoglint` package (e.g., to import it in your own code), you can install it directly:

```bash
pip install git+https://github.com/knmoriwaki/cosmoglint.git
```


Several libraries needs to be additionally installed to run the scripts and notebooks:
```bash
pip install -r requirements.txt
```

## Training 

Example:
```bash
python train_transformer.py --data_path [data_path] --norm_param_file [norm_param_file] 
```

Options:
- `--data_path`: Path(s) to the training data. Data is an hdf5 file that contains properties of halos and galaxies. In addition to those for input and output features, the number of galaxies in each halo (`GroupNsubs`) should be provided. Multiple files can be passed.
- `--norm_param_file`: Path to the json file that specifies the normalization settings. Each key (e.g., `HaloMass`) maps to a dictionary with `min` / `max` and `norm`. If `norm` is `"log"` or `"log_with_sign"`, the `min` / `max` normalization is applied after the log conversion.
- `--input_features`: List of the input properties (default: `["GroupMass"]`)
- `--output_features`: List of the output properties (default: `["SubhaloSFR", "SubhaloDist", "SubhaloVrad", "SubhaloVtan"]`)
- `--max_length`: Maximum number of galaxies (sequence length) per halo (default: 30).
- `--use_flat_representation`: If true, use flattened point features (B, N * M). If false, keep (B, N, M). Use `--no-use_flat_representation` to set it to false (default: true).

## Create mock data cube

Example:
```bash
python create_data_cube.py --input_fname [input_fname] --model_dir [model_dir] 
```

Options:
- `--input_fname`: Path to the halo catalog. Text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns) and catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format are supported.
- `--model_dir`: Path to a directory containing the trained model (`model.pth` and `args.json`). If not set, column 7 of the input file is used as intensity.
- `--boxsize`: Size of the simulation box in comoving units [Mpc/h] (default: 100.0).
- `--redshift_space`: If set, positions are converted to redshift space using halo velocities.
- `--gen_both`: If set, generates both real-space and redshift-space data cubes.
- `--npix`: Number of pixels in the x and y directions for the data cube (default: 100).
- `--npix_z`: Number of pixels in the z direction (default: 90).

## Create lightcone

Example:
```bash
python create_lightcone.py --input_fname [input_fname] --model_dir [model_dir] --model_config_file [model_config_file]
```

Example of `model_config_file`:
```json
{
  "33": ["transformer1_33_ep40_bs512_w0.02", 2.002],
  "21": ["transformer1_21_ep60_bs512_w0.02", 4.008]
}
```

- `--input_fname`: Path to the lightcone halo catalog. Pinocchio format is supported.
- `--output_fname`: Output filename (HDF5 format).
- `--model_dir`: Path to a directory containing the trained models. 
- `--model_config_file`: Path to a JSON file that contains the names of the trained models to be used for each redshift bin. The JSON file is a dictionary where each key is a stringified snapshot ID, and the value is a list containing the model directory relative to `model_dir` and the redshift.
- `--redshift_space`: If set, generate output in redshift space.
- `redshift_min`, `--redshift_max`: Redshift range for the lightcone.
- `dz`: Redshift bin width. Indicates dlogz if `--use_logz` is given.
- `use_logz`: Use dlogz instead of dz for redshift binning.
- `--side_length`, `--angular_resolution`: Angular size and resolution (arcsec) of the simulated map.
- `--gen_catalog`: If set, generate a galaxy catalog with SFR greater than --catalog_threshold.
- `--catalog_threshold`: SFR threshold for inclusion in the catalog.

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `plot_transformer.ipynb`: visualize training results
- `plot_data_cube.ipynb`: visualize created data cube
- `plot_lightcone.ipynb`: visualize lightcone data


## Citation

If you use CosmoGLINT in your research, please cite [Moriwaki et al. 2025](https://arxiv.org/abs/2506.16843)

```
@ARTICLE{CosmoGLINT,
  title = {CosmoGLINT: Cosmological Generative Model for Line Intensity Mapping with Transformer},
  author = {{Moriwaki}, Kana and {Jun}, Rui Lan and {Osato}, Ken and {Yoshida}, Naoki},
  journal = {arXiv preprints},
  year = 2025,
  month = jun,
  eid = {arXiv:2506.16843},
  doi = {10.48550/arXiv.2506.16843},
  archivePrefix = {arXiv},
  eprint = {2506.16843},
  primaryClass = {astro-ph.CO}
}
```