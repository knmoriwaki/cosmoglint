# CosmoGLINT: Cosmological Generative model for Line INtensity mapping with Transformer

Transformer-based models that generate galaxies.

---

## Installation

This package is available on PyPI.

```bash
pip install cosmoglint
```

This package requires PyTorch.
Please install PyTorch first following https://pytorch.org

## Documentation & Source code

GitHub: https://github.com/knmoriwaki/cosmoglint

## Basic usage

Load model:
```python
import json
from cosmoglint.model.transformer import transformer_model

with open(f"args.json", "r") as f:
  option = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
  
model = transformer_model(option)
```
with a json file (example):
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
| **`model_name`**       | Name or identifier for the model configuration (default: `"transformer1"`). Available options are:<br> - `"transformer1"`: halo is prepended to the sequence<br> - `"transformer2"`: halo and galaxy features are embedded together | |
| **`max_length`**       | Maximum number of galaxies (sequence length) the model will process per halo.|
| **`d_model`**          | Dimensionality of the internal feature space (i.e., size of the token embeddings and hidden layers in the transformer).                             |
| **`num_layers`**       | Number of transformer decoder layers stacked in the model.                                          |
| **`num_heads`**        | Number of attention heads in the multi-head self-attention layers.                                       |
| **`num_features_out`** | Total number of output bins for the probability distribution.  |
| **`num_features_in`**  | Number of features per galaxy (e.g., SFR, relative distance, radial/tangential velocity).     |


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