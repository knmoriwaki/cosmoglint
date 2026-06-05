# Documentation for CosmoGLINT

## Overview

This repository includes:

- A package of Transformer-based models that generate galaxy properties from halo properties of density map.
- Scripts for training and mock catalog generation.
- Example notebooks for result visualization.

~~Models trained with TNG300-1 at z = 0.5 - 6 and generated data are available at [Google Drive](https://drive.google.com/drive/folders/1IFje9tNRf4Dr3NufqzlDdGMFTEDpsm35?usp=share_link).~~

Pre-trained model for the new version will be provided soon.

---

## Installation

This package requires PyTorch>=3.9. 
Please install PyTorch first following https://pytorch.org

Install package:

```bash
git clone https://github.com/knmoriwaki/cosmoglint.git
cd cosmoglint
pip install .
```

For developer (editable mode):

```bash
pip install -e .
```

If you only need the `cosmoglint` package (e.g., to import it in your own code), you can install it directly:

```bash
pip install git+https://github.com/knmoriwaki/cosmoglint.git
```

To use scripts and notebooks, install additional libraries:
```bash
pip install -r requirements.txt
```


## Model Usage

### Load model:
```python
from cosmoglint.model.transformer import Transformer1

cfg = {
  "max_length": 50, 
  "d_model": 128, 
  "num_layers": 4, 
  "num_heads": 8, 
  "num_features_cond": 1, 
  "num_features_in": 4, 
  "num_features_out": 100
}

model = Transformer1(**cfg)
```

### Predict probability:
```python
prob = model(condition, seq) 
```

### Generate new galaxies:
```python
generated, prob = model.generate(condition, seq=seq, prob_threshold=1e-5)
```

### Input:
- `condition`: Conditioning input passed to the model. The shape of `condition` depends on the model being used. See the [Models](#models) section below for details.
- `seq`: a tensor of shape `(B, L, C_g)`, containing the properties of up to `L` galaxies for each of the `N` halos in the batch. Each feature vector of size `C_g` may include, for example, the halo mass, relative distance to the halo center, radial velocity, and tangential velocity. Set to `None` to generate galaxies from scratch.
- `prob_threshold` (optional): when sampling, the probability below this threshold is set to zero.

### Output:
- `prob`: a tensor of shape `(B, L, C_g, d)`. `prob[i,j,k,:]` is the probability distribution over `d` bins for the k-th parameter of the **(j+1)-th galaxy** in the sequence for the i-th batch element. 
- `generated`: a tensor of shape `(B, L, C_g)`. `generated[i,j,k]` is the sampled values for each parameter of **(j+1)-th galaxy** in the sequence for the i-th batch element.

### Shape: 
- `B`: Batch size 
- `L`: Sequence length 
- `C_h`: Number of halo properties 
- `C_g`: Number of galaxy properties predicted 
- `d`: Number of bins for the probability distribution of each parameter

### Models:
| Class                    | Description                                                                                                                                                                |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`Transformer1`**       | halo is prepended to the sequence. `condition` should be a tensor of shape `(B, C_h)`, containing the halo properties.  |
| **`Transformer2`** | halo and galaxy features are embedded together. `condition` should be a tensor of shape `(B, C_h)`, containing the halo properties. |
| **`MeshConditionedTransformer`** | 3d mesh data is encoded in a sequence and decoded with the target sequence. `condition` should be a tensor of shape `(B, C_h, N, N, N)` |
| **`MeshSequenceConditionedTransformer`** |  3d mesh and context sequence data is encoded in a sequence and decoded with the target sequence. `condition` should be a dict including "mesh" `(B, C_h, N, N, N)`, "context" `(B, L_ctx, C_g)`, "mask_ctx" `(B, L_ctx)`, "boundary" `(B, 6)`. |

### Options:
| Key                    | Description                                                                                                                                                                |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`max_length`**       | Maximum number of galaxies (sequence length) the model will process per halo.|
| **`d_model`**          | Dimensionality of the internal feature space (i.e., size of the token embeddings and hidden layers in the transformer).                             |
| **`num_layers`**       | Number of transformer decoder layers stacked in the model.                                          |
| **`num_heads`**        | Number of attention heads in the multi-head self-attention layers.                                       |
| **`num_features_cond`** | NUmber of features per halo (e.g., halo mass)                 |
| **`num_features_out`** | Total number of output bins for the probability distribution.  |
| **`num_features_in`**  | Number of features per galaxy (e.g., SFR, relative distance, radial/tangential velocity).     |

---


## Training scripts

Example:
```bash
cd scripts
python train_transformer.py --config_file [config_file] 
```

The config file is a YAML file that specifies the details of the dataset and the model. The example config files are located in `scripts/config`.

### Model-related fields (config file):
- `model_name`: Name of the model architecture to use (default: "transformer1"). 
- `d_model`: Dimensionality of the transformer’s internal feature representation (default: 128).
- `num_layers`: Number of transformer encoder layers (default: 4).
- `num_heads`: Number of attention heads in each multi-head attention layer (default: 8).
- `num_features_out`: Total number of output bins across all predicted parameters. Typically C × d, where C is the number of output features and d is the number of bins per parameter.

### Data-related fields (config file):
- `data_path`: Path(s) to the training data. Data is an hdf5 file that contains properties of halos and galaxies. In addition to those for input and output features, the number of galaxies in each halo (`Group/GroupNsubs`) should be provided. Multiple files can be passed.
- `data_path_mesh`: Path(s) to the mesh data. Required when using "mesh_conditioned_transformer" or "mesh_sequence_conditioned_transformer".
- `global_param_file`: Path to the global parameters file(s). The header should include `global_features`. (default: None)
- `indices`: If the data path contains `*` (e.g., `.../run_*`), it will be expanded by replacing `*` with integers in the specified range (e.g., `0–999`).  
- `input_features`: List of the input properties (default: `["GroupMass"]`)
- `output_features`: List of the output properties (default: `["SubhaloSFR", "SubhaloDist", "SubhaloVrad", "SubhaloVtan"]`)
- `global_features`: List of global properties. If not None, `global_param_file` should be provided (default: None)
- `norm_param_file`: Path to the json file that specifies the normalization settings. Each key (e.g., `GroupMass`) maps to a dictionary with `min` / `max` and `norm`. If `norm` is `"log"` or `"log_with_sign"`, the `min` / `max` normalization is applied after the log conversion.
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

### Command-line options:
- `--gpu_id`: ID of the GPU to use (default: "0"). Accepts string values like "0", "1", etc.
- `--seed`: Random seed for reproducibility (default: 12345).
- `--show_pbar`: Show progress bar. Use `--no-show_pbar` to disable progress bar. (default: True)
- `--output_dir`: Directory where outputs (e.g., model checkpoints, logs) will be saved (default: "output").
- `--train_ratio`: Fraction of the data to use for training (the rest is used for validation). Default is 0.9.
- `--exclude_ratio`: The cubic region whose side length is `BoxSize` multiplied by this ratio is not used for training. `BoxSize` and `GroupPos` should be provided in the data file if a positive ratio is set.
- `--batch_size`: Number of halo sequences per batch (default: 128).
- `--num_epochs`: Number of training epochs (default: 2).
- `--lr`: Learning rate for the optimizer (default: 1e-3).
- `--dropout`: Dropout rate used in the model (default: 0.0).
- `--sampler_weight_min`: Minimum weight for the sampler. Set to < 1 to use a sampler that balances the training data based on halo's primary property (e.g., halo mass), otherwise the sampler is not used. 
- `--save_freq`: Frequency (in epochs) at which the model is saved during training (default: 100).

---

## Create scripts

Example (from halo)
```bash
cd scripts
python create_dc.py --input_fname [input_fname] --model_dir [model_dir] 
```
Example (from mesh)
```bash
cd scripts
python create_dc_mesh.py --input_fname [input_fname] --model_dir [model_dir] 
```

### Important options:
- `--input_fname`: Path to the halo catalog. The following file formats are supported:
  - HDF5 file in gadget format.
  - Text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns)
  - [Pinocchio](https://github.com/pigimonaco/Pinocchio) format
- `--output_fname`: Name of the output hdf5 file to write the generated map to (default: None).
- `--output_catalog_fname`: Name of the output file to write the generated catalog to. File is ASCII for `create_dc.py` and hdf5 for `create_dc_mesh.py` (default: None).

- `--model_dir`: Path to a directory containing the trained model (`model.pth` and `args.json`). If not set, column 7 of the input file is used as intensity.
- `--boxsize`: Size of the simulation box in comoving units [Mpc/h] (default: 100.0).
- `--redshift_space`: If set, generate output in redshift space in addition to output in real space.
- `--gen_both`: If set, generates both real-space and redshift-space data cubes.
- `--npix`: Number of pixels in the x and y directions for the data cube (default: 100).
- `--npix_z`: Number of pixels in the z direction (default: 90).

### Other options:
- `--gpu_id`: GPU ID to use (default: 0).
- `--seed`: Random seed for reproducibility (default: 12345).
- `--catalog_threshold`: Minimum star formation rate (SFR) [Msun/yr] for galaxies to be included in the catalog (default: 10).
- `--logm_min`: Minimum log halo mass [Msun] to be included in the mock (default: 11.0).
- `--threshold`: Only galaxies with SFR > threshold [Msun/yr] will be used in the mock (default: 1e-3).
- `--mass_correction_factor`: Multiplier applied to halo mass before galaxy generation (default: 1.0). Useful if calibration is needed.
- `--max_sfr_file`: File containing maximum normalized SFR values for each halo mass bin (default: None).


---

## Lightcone creation scripts

One can also create a mock lightcone data. This requires models trained on multiple redshift.

Example:
```bash
cd scripts
python create_lc.py --input_fname [input_fname] --model_dir [model_dir] --model_config_file [model_config_file]
```

Example of `model_config_file`:
```json
{
  "33": ["transformer1_33_ep40_bs512_w0.02", 2.002],
  "21": ["transformer1_21_ep60_bs512_w0.02", 4.008]
}
```

### Important options:
- `--input_fname`: Path to the lightcone halo catalog. Pinocchio format is supported.
- `--output_fname`: Name of the output hdf5 file to write the generated intensity map to (default: None).
- `--output_catalog fname`: Name of the output file to write the generated catalog to (default: None).

- `--model_dir`: Path to a directory containing the trained models. 
- `--model_config_file`: Path to a JSON file that contains the names of the trained models to be used for each redshift bin. The JSON file is a dictionary where each key is a stringified snapshot ID, and the value is a list containing the model directory relative to `model_dir` and the redshift.
- `--redshift_space`: If set, generate output in redshift space in addition to output in real space.
- `redshift_min`, `--redshift_max`: Redshift range for the lightcone.
- `--side_length`: Angular size of the simulated map in arcsec (default: 300).

### Options for catalog 
The following options are required if `output_catalog_fname` is defined
- `--catalog_threshold`: SFR threshold for inclusion in the catalog.

### Options for intensity map 
The following options are required if `output_fname` is defined
- `--line_list`: List of line names (default `["[CII]"]`)
- `--angular_resolution`: Angular resolution in arcsec. (default: 30)
- `--fmin`: Minimum frequency in GHz (default: 10)
- `--fmax`: Maximum frequency in GHz (default: 100)
- `--intensity_unit`: Intensity unit to use. Available options: "Jy/sr", "erg/s/cm2/Hz/beam", "erg/s/cm2/sr" (default: "Jy/sr")
- `--sigma`: Log-normal scatter [dex] added to the luminosity–SFR relation (default: 0.2)

### Other options:
- `--gpu_id`: GPU ID to use (default: 0).
- `--seed`: Random seed for reproducibility (default: 12345).
- `--param_dir`: Path to a directory containing the file of maximum normalized SFR for each mass bin (default: None).
- `--redshift_min`, `--redshift_max`: Minimum and maximum redshift range for the mock data.
- `--logm_min`: Minimum log halo mass for selecting galaxies.
- `--threshold`: Minimum SFR threshold for emission line generation.
- `--mass_correction_factor`: Multiplier applied to halo mass before galaxy generation (default: 1.0). Useful if calibration is needed.


---

## Notebooks

- `quick_check_halo.ipynb`, `quick_check_mesh.ipynb`: For quick look at training results 
- `gen_analysis_halo.ipynb`, `gen_analysis_mesh.ipynb`: Visualize and analyse created data
- `gen_analysis_lightcone_halo.ipynb`: Visualize and analyse created light cone data

## Other models

- `transformer`: Default model. Transformer outputs one-hot vectors that represent the probability distributions of parameters.
- `transformer_nf`: Transformer + normalizing flow model. NF samples galaxies conditioned on the output of Transformer. Note that a package `nflows` is additionally required for using this model.


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