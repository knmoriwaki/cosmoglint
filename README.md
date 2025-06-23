# CosmoGLINT: Cosmological Generative model for Line INtensity mapping with Transformer

This repository includes:

- cosmoglint, a package of Transformer-based models that generate galaxy properties from halo mass.
- Scripts for training and mock catalog generation.
- Example notebooks for result visualization.

Models trained with TNG300-1 at z = 0.5 - 6 and generated data are available at [Google Drive](https://drive.google.com/drive/folders/1IFje9tNRf4Dr3NufqzlDdGMFTEDpsm35?usp=share_link).

For detailed usage and options, see [DOCUMENTATION](./DOCUMENTATION.md).

---

## Installation

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