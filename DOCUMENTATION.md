# Documentation for Transformer-based Mock Generator for Line Intensity Mapping


## Overview

This repository includes:

- A package of Transformer-based models that generate galaxy properties from halo mass.
- Scripts for training and mock catalog generation.
- Example notebooks for result visualization.

Pre-trained models and example datasets available at: [Google Drive](https://drive.google.com/drive/folders/1IFje9tNRf4Dr3NufqzlDdGMFTEDpsm35?usp=share_link).


---

## Model

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

For developer (editable mode):

```bash
pip install -e .
```

---

Load model:
```python
import json
from lim-mock-generator.model.transformer import my_model

with open(f"args.json", "r") as f:
  option = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
  
model = my_model(option)
```
with a json file:
```json
{"model_name": "transformer1", "max_length": 50, "d_model": 128, "num_layers": 4, "num_heads": 8, "num_features_out": 100, "num_features_in": 4}
```

Predict probability:
```python
prob = model(context, x) 
```

Generate new galaxies:
```python
generated, prob = model.generate(context, x, prob_threshold=1e-5)
```

Input:
- `context`: a tensor of shape `(N, C_h)`, containing the properties of halo.
- `x`: a tensor of shape `(N, L, C_g)`, containing the properties of up to `L` galaxies for each of the `N` halos in the batch. Each feature vector of size `C_g` may include, for example, the halo mass, relative distance to the halo center, radial velocity, and tangential velocity. Input `None` to generate from scratch.
- `prob_threshold` (optional): when sampling, the probability below this threshold is set to zero.

Output:
- `prob`: a tensor of shape `(N, L, C_g, d)`. `prob[i,j,k,:]` is the probability distribution over `d` bins for the k-th parameter of the **(j+1)-th galaxy** in the sequence for the i-th batch element. 
- `generated`: a tensor of shape `(N, L, C_g)`. `generated[i,j,k]` is the sampled values for each parameter of **(j+1)-th galaxy** in the sequence for the i-th batch element.

Shape: 
- `N`: Batch size 
- `L`: Sequence length 
- `C_h`: Number of halo properties 
- `C_g`: Number of galaxy properties predicted 
- `d`: Number of bins for the probability distribution of each parameter

Options:
| Key                    | Description                                                                                                                                                                |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`model_name`**       | Name or identifier for the model configuration (default: `"transformer1"`). Available options are:<br> - `"transformer1"`: halo is prepended to the sequence; positional encoding is concatenated separately<br> - `"transformer2"`: halo and fixed position information are embedded together with galaxy features<br> - `"transformer3"`: halo and galaxy features are embedded together; positional encoding is concatenated separately | |
| **`max_length`**       | Maximum number of galaxies (sequence length) the model will process per halo.|
| **`d_model`**          | Dimensionality of the internal feature space (i.e., size of the token embeddings and hidden layers in the transformer).                             |
| **`num_layers`**       | Number of transformer decoder layers stacked in the model.                                          |
| **`num_heads`**        | Number of attention heads in the multi-head self-attention layers.                                       |
| **`num_features_out`** | Total number of output bins for the probability distribution.  |
| **`num_features_in`**  | Number of features per galaxy (e.g., halo mass, relative distance, radial/tangential velocity).     |


---

## Scripts and notebook

The scripts in `scripts` can be used for training and mock generation. Notebooks in `notebooks` can be used for visualization.

To use scripts and notebooks, install additional libraries:
```bash
pip install -r requirements.txt
```

### Training 

Example:
```bash
cd scripts
python train_transformer.py --data_path [data_path] --norm_param_file [norm_param_file] --use_dist --use_vel
```

Important options:
- `--data_path`: Path(s) to the training data. Data is an hdf5 file that contains properties of halos (`HaloMass`, `NumSubgroups`, `Offset`) and galaxies (`SubgroupSFR`, `SubgroupDist`, `SubgroupVrad`, `SubgroupVtan`), where `NumSubgroups` and `Offset` are used for determining host halos of galaxies. Multiple files can be passed.
- `--norm_param_file`: Path to the normalization parameter file. The file contains minimum (1st column) and maximum (2nd column) values for the input and output parameters.
- `--use_dist`: If set, the distance to halo is predicted
- `--use_vel`: If set, relative velocity to halo is predicted
- `--max_length`: Maximum number of galaxies (sequence length) per halo (default: 30).

Other options:
- `--gpu_id`: ID of the GPU to use (default: "0"). Accepts string values like "0", "1", etc.
- `--seed`: Random seed for reproducibility (default: 12345).
- `--output_dir`: Directory where outputs (e.g., model checkpoints, logs) will be saved (default: "output").

- `--train_ratio`: Fraction of the data to use for training (the rest is used for validation). Default is 0.9.
- `--batch_size`: Number of halo sequences per batch (default: 128).
- `--num_epochs`: Number of training epochs (default: 2).
- `--lr`: Learning rate for the optimizer (default: 1e-3).
- `--dropout`: Dropout rate used in the model (default: 0.0).
- `--sampler_weight_min`: Minimum weight for the sampler. Set to < 1 to use a sampler that balances the training data based on halo's primary property (e.g., halo mass), otherwise the sampler is not used. 
- `--lambda_penalty_loss`: Coefficient for the penalty loss.
- `--save_freq`: Frequency (in epochs) at which the model is saved during training (default: 100).

- `--model_name`: Name of the model architecture to use (default: "transformer1"). 
- `--d_model`: Dimensionality of the transformer’s internal feature representation (default: 128).
- `--num_layers`: Number of transformer encoder layers (default: 4).
- `--num_heads`: Number of attention heads in each multi-head attention layer (default: 8).
- `--num_features_out`: Total number of output bins across all predicted parameters. Typically C × d, where C is the number of output features and d is the number of bins per parameter.

Note: `num_features_in` is automatically determined from the shape of dataset

---

### Create data cube

Example:
```bash
cd scripts
python create_data_cube.py --input_fname [input_fname] --model_dir [model_dir] 
```

Important options:
- `--input_fname`: Path to the halo catalog. Text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns) and catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format are supported.
- `--model_dir`: Path to a directory containing the trained model (`model.pth` and `args.json`). If not set, column 7 of the input file is used as intensity.
- `--boxsize`: Size of the simulation box in comoving units [Mpc/h] (default: 100.0).
- `--redshift_space`: If set, positions are converted to redshift space using halo velocities.
- `--gen_both`: If set, generates both real-space and redshift-space data cubes.
- `--npix`: Number of pixels in the x and y directions for the data cube (default: 100).
- `--npix_z`: Number of pixels in the z direction (default: 90).

Other options:
- `--gpu_id`: GPU ID to use (default: 0).
- `--seed`: Random seed for reproducibility (default: 12345).
- `--NN_model_dir`: Path to a directory containing a second-stage neural network model. Only if both `--model_dir` and `--NN_model_dir` are given, galaxies are generated via a two-step method.
- `--output_fname`: Name of the output file to write the generated data to (default: test.h5).
- `--gen_catalog`: If set, outputs a galaxy catalog instead of a 3D data cube.
- `--catalog_threshold`: Minimum star formation rate (SFR) [Msun/yr] for galaxies to be included in the catalog (default: 10).
- `--logm_min`: Minimum log halo mass [Msun] to be included in the mock (default: 11.0).
- `--threshold`: Only galaxies with SFR > threshold [Msun/yr] will be used in the mock (default: 1e-3).
- `--mass_correction_factor`: Multiplier applied to halo mass before galaxy generation (default: 1.0). Useful if calibration is needed.
- `--max_ids_file`: File containing maximum IDs for SFR

---

### Create mock data

After training with multiple redshift data, you can create a mock data.

Example:
```bash
cd scripts_lightcone
python create_lightcone.py --input_fname [input_fname] --model_dir [model_dir]
```

Important options:
- `--input_fname`: Path to the lightcone data. Pinocchio format is supported.
- `--output_fname`: Output filename (HDF5 format).
- `--model_dir`: Path to a directory containing the directories of trained models. The names of the directories to be used for each redshift bin is hardcoded in `lightcone_utils.py`.
- `--redshift_space`: If set, generate output in redshift space.
- `--side_length`, `--angular_resolution`: Angular size and resolution (arcsec) of the simulated map.
- `--fmin`,`--fmax`: Frequency range [GHz] for the mock cube.
- `--R`: Spectral resolution 

Other options:
- `--seed`: Random seed for reproducibility (default: 12345).
- `--NN_model_dir`: Path to the directory containing the neural network model for intensity prediction.
- `--sigma`: Log-normal scatter [dex] added to the luminosity–SFR relation.
- `--gen_both`: If set, generate both real and redshift space data.
- `--redshift_min`, --redshift_max: Minimum and maximum redshift range for the mock data.
- `--logm_min`: Minimum log halo mass for selecting galaxies.
- `--threshold`: Minimum SFR threshold for emission line generation.
- `--gen_catalog`: If set, generate a galaxy catalog with SFR greater than --catalog_threshold.
- `--catalog_threshold`: SFR threshold for inclusion in the catalog.
- `--max_ids_file`: File containing maximum IDs for SFR

---

## Notebooks

- `plot_transformer.ipynb`: visualize training results
- `plot_data_cube.ipynb`: visualize created data cube
- `plot_lightcone.ipynb`: visualize created light cone mock

## Other models

- `transformer`: Default model. Transformer outputs one-hot vectors that represent the probability distributions of parameters.
- `transformer_nf`: A normalizing flow samples galaxies conditioned on the output of Transformer.
