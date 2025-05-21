# LIM Mock Generator

A Python package of generative models for line intensity mapping (LIM) experiments. 

A pre-trained model trained with TNG300-1 at z = 2 as well as example inputs for training and mock creation are available at [Google Drive](https://drive.google.com/drive/folders/1HRkRdfti8XaIPyF3er5QJmFX3WXCmAQI?usp=sharing).

Currently, the package is designed to work with comoving volume box data only. Support for lightcone data may be added in a future update.

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


Several libraries in `requirements.txt` needs to be additionally installed to use the scripts and notebooks.

## Usage

```
from lim_mock_generator.model.transformer import my_model

with open("./examples/args_transformer.json".format(model_dir), "r") as f:
    opt = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

model = my_model(opt)

nsample = 10
x = torch.rand(nsample)
generated, prob = model.generate(x)
```

## Scripts

### Training 

Put training data at `[data_path]` and a file that contains normalization parameters at `[norm_param_file]`, and run the following:
```
cd ./scripts
python train_transformer.py --data_path [data_path] --norm_param_file [norm_param_file] --use_dist --use_vel
```
The distance and velocity relative to halo are modeled when options `--use_dist` and `--use_vel` are given.

The training data should be a hdf5 file that contains the following properties of halos:
- `HaloMass` 
- `NumSubgroups` 
- `Offset` 

and the following properties of galaxies:
- `SubgroupSFR` 
- `SubgroupDist` 
- `SubgroupVrad` 
- `SubgroupVtan` 

`NumSubgroups` and `Offset` are used for determining host halos of galaxies.

The normalization parameter file should be a ascii text file that contains minimum (1st column) and maximum (2nd column) values for the input and output parameters.

### Create mock data

Put a halo catalog at `[input_fname]` and the trained model (model.pth and args.json) at `[model_dir]` and run the following.
```
cd ./scripts
python create_data.py --input_fname [input_fname] --model_dir [model_dir]
```

The halo catalog should be a text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns).

Halo catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format is also supported.

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `plot_transformer.ipynb`: visualize training results
- `plot_mock.ipynb`: visualize created mock data


## Citation

