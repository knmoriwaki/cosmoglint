# CosmoGLINT: Cosmological Generative model for Line INtensity mapping with Transformer

This repository includes:

- cosmoglint, a package of Transformer-based models that generate galaxy properties from halo properties or density map.
- Scripts for training and mock catalog generation.
- Example notebooks for result visualization.

For detailed usage and options, see [DOCUMENTATION](./DOCUMENTATION.md).

---

## Installation

Python>=3.9 is required. 

This package requires PyTorch.
Please install PyTorch first following https://pytorch.org

Install package:

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
cd scripts
python train_transformer.py --config_file [config_file] 
```

The config file is a YAML file that specifies the details of the dataset and the model.

### Model-related fields (config file):
- `model_name`: Name of the model architecture to use (default: "transformer1"). 
- `d_model`: Dimensionality of the transformer’s internal feature representation (default: 128).
- `num_layers`: Number of transformer encoder layers (default: 4).
- `num_heads`: Number of attention heads in each multi-head attention layer (default: 8).
- `num_features_out`: Total number of output bins across all predicted parameters. Typically C × d, where C is the number of output features and d is the number of bins per parameter.

### Data-related fields (config file):
- `data_path`: Path(s) to the training data. Data is an hdf5 file that contains properties of halos and galaxies. In addition to those for input and output features, the number of galaxies in each halo (`GroupNsubs`) should be provided. Multiple files can be passed.
- `data_path_mesh`: Path(s) to the mesh data. Required when using "mesh_conditioned_transformer" or "mesh_sequence_conditioned_transformer".
- `global_param_file`: Path to the global parameters file(s). The header should include `global_features`. (default: None)
- `indices`: If the data path contains `*` (e.g., `.../run_*`), it will be expanded by replacing `*` with integers in the specified range (e.g., `0–999`).  
- `input_features`: List of the input properties (default: `["GroupMass"]`)
- `output_features`: List of the output properties (default: `["SubhaloSFR", "SubhaloDist", "SubhaloVrad", "SubhaloVtan"]`)
- `global_features`: List of global properties. If not None, `global_param_file` should be provided (default: None)
- `norm_param_file`: Path to the json file that specifies the normalization settings. Each key (e.g., `HaloMass`) maps to a dictionary with `min` / `max` and `norm`. If `norm` is `"log"` or `"log_with_sign"`, the `min` / `max` normalization is applied after the log conversion.
Example `norm_param_file`:

  ```json
  {
    "GroupMass": {
      "min": 1.0,
      "max": 5.0,
      "norm": "log"
    },
    "SubhaloSFR": {
      "min": -3.0,
      "max": 3.0,
      "norm": "log"
    }
  }
  ```
- `max_length`: Maximum number of galaxies (sequence length) per halo (default: 30).
- `use_flat_representation`: If true, use flattened point features (B, N * M). If false, keep (B, N, M). Set this to `true` when you want to model correlations among multiple parameters. (default: false)


## Create mock data cube

Example:
```bash
python create_data_cube.py --input_fname [input_fname] --model_dir [model_dir] 
```

### Options:
- `--input_fname`: Path to the halo catalog. Text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns) and catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format are supported.
- `--output_fname`: Name of the output hdf5 file to write the generated map to (default: None).
- `--output_catalog_fname`: Name of the output hdf5 file to write the generated catalog to (default: None).

- `--model_dir`: Path to a directory containing the trained model (`model.pth` and `args.json`). If not set, column 7 of the input file is used as intensity.
- `--boxsize`: Size of the simulation box in comoving units [Mpc/h] (default: 100.0).
- `--redshift_space`: If set, generate output in redshift space in addition to output in real space.
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

### Options: 
- `--input_fname`: Path to the lightcone halo catalog. Pinocchio format is supported.
- `--output_fname`: Name of the output hdf5 file to write the generated map to (default: None).
- `--output_catalog_fname`: Name of the output hdf5 file to write the generated catalog to (default: None).

- `--model_dir`: Path to a directory containing the trained models. 
- `--model_config_file`: Path to a JSON file that contains the names of the trained models to be used for each redshift bin. The JSON file is a dictionary where each key is a stringified snapshot ID, and the value is a list containing the model directory relative to `model_dir` and the redshift.
- `--redshift_space`: If set, generate output in redshift space in addition to output in real space.
- `redshift_min`, `--redshift_max`: Redshift range for the lightcone.
- `--side_length`: Angular size of the simulated map in arcsec (default: 300).

### Options for catalog 
The following options are required if `output_catalog_fname` is defined
- `--catalog_threshold`: SFR threshold for inclusion in the catalog.

### Options for intensity map 
The following options are required if if `output_fname` is defined
- `--line_list`: List of line names (default `["[CII]"]`)
- `--angular_resolution`: Angular resolution in arcsec. (default: 30)
- `--fmin`: Minimum frequency in GHz (default: 10)
- `--fmax`: Maximum frequency in GHz (default: 100)
- `--intensity_unit`: Intensity unit to use. Available options: "Jy/sr", "erg/s/cm2/Hz/beam", "erg/s/cm2/sr" (default: "Jy/sr")

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `quick_check_halo.ipynb`, `quick_check_mesh.ipynb`: For quick look at training results 
- `gen_analysis_halo.ipynb`, `gen_analysis_mesh.ipynb`: Visualize and analyse created data
- `gen_analysis_lightcone_halo.ipynb`: Visualize and analyse created light cone data

## Citation

If you use CosmoGLINT in your research, please cite [Moriwaki et al. 2026](https://arxiv.org/abs/2506.16843)

```
@ARTICLE{CosmoGLINT,
  title = {CosmoGLINT: Cosmological Generative Model for Line Intensity Mapping with Transformer},
  author = {{Moriwaki}, Kana and {Jun}, Rui Lan and {Osato}, Ken and {Yoshida}, Naoki},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year = 2026,
  month = jan,
  volume = {545},
  number = {3},
  eid = {staf2124},
  pages = {staf2124},
  doi = {10.1093/mnras/staf2124},
  archivePrefix = {arXiv},
  eprint = {2506.16843},
  primaryClass = {astro-ph.CO}
}
```