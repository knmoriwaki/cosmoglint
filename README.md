# LIM Mock Generator

A Python package of transformer-based generative models for line intensity mapping (LIM) experiments. 

A model trained with TNG300-1 at z = 2 as well as example inputs for training and mock creation are available at [Google Drive](https://drive.google.com/drive/folders/1HRkRdfti8XaIPyF3er5QJmFX3WXCmAQI?usp=sharing).

Currently, the package is designed to work with comoving volume box data only. Support for lightcone data may be added in a future update.

---

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/knmoriwaki/lim-mock-generator.git
```

Or, if you cloned the repository locally:

```bash
git clone https://github.com/knmoriwaki/lim-mock-generator.git
cd lim-mock-generator
pip install -e .
```

To use scripts and notebooks, several libraries needs to be additionally installed:
```bash
pip install -r requirements.txt
```

## Training 

Run the following command:
```bash
cd ./scripts
python train_transformer.py --data_path [data_path] --norm_param_file [norm_param_file] --use_dist --use_vel
```

Options:
- `[data_path]`: path to the training data. Data is an hdf5 file that contains properties of halos (`HaloMass`, `NumSubgroups`, `Offset`) and galaxies (`SubgroupSFR`, `SubgroupDist`, `SubgroupVrad`, `SubgroupVtan`), where `NumSubgroups` and `Offset` are used for determining host halos of galaxies.
- `[norm_param_file]`: normalization parameter file name. The file contains minimum (1st column) and maximum (2nd column) values for the input and output parameters.
- `--use_dist`: distance to halo is modeled if this is given.
- `--use_vel`: relative velocity to halo is modeled if this is given.

## Create mock data

Run the following command:
```bash
cd ./scripts
python create_data.py --model_dir [model_dir] --input_fname [input_fname] 
```
Options:
- `[model_dir]`: directory containing `model.pth` and `args.json`.
- `[input_fname]`: path to the halo catalog. Text file that contains halo mass [Msun] in log scale (1st column), comving positions [Mpc/h] (2nd to 4th columns), and velocities [km/s] (5th to 8th columns) and catalog in [Pinocchio](https://github.com/pigimonaco/Pinocchio) format are supported.

## Visualization

Example Jupyter notebooks are available in the `notebooks/` directory:

- `plot_transformer.ipynb`: visualize training results
- `plot_mock.ipynb`: visualize created mock data


## Citation

