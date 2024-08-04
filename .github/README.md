<a href="https://colab.research.google.com/github/SauravMaheshkar/samv2/blob/main/examples/notebooks/samv2_prompted_segmentation_with_wandb_tables.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![Build and Tests](https://github.com/SauravMaheshkar/samv2/actions/workflows/ci.yml/badge.svg)](https://github.com/SauravMaheshkar/samv2/actions/workflows/ci.yml)

CPU **compatible** fork of the official SAMv2 implementation.

## Features 🚀

* CPU compatible
* ships with config files
* Run image and video inference on CPUs
* [Example notebooks](../examples/notebooks/) showcasing inference using weights and biases.

## Installation

You can download it from [pypi](https://pypi.org/) using `pip` as follows:

```bash
pip install samv2
```

or from the repository:

```bash
pip install git+https://github.com/SauravMaheshkar/samv2.git
```

## Usage

After downloading the official weights, you can use the `load_model()` helper method to instantiate a model.

```python
from sam2 import load_model

model = load_model(
    variant="tiny",
    ckpt_path="artifacts/sam2_hiera_tiny.pt",
    device="cpu"
)
```

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SauravMaheshkar/samv2/blob/main/examples/notebooks/samv2_prompted_segmentation_with_wandb_tables.ipynb) Example Notebook to run prompted segmentation on images logging predictions as W&B Tables.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SauravMaheshkar/samv2/blob/main/examples/notebooks/samv2_automatic_segmentation_with_wandb_tables.ipynb) Example Notebook to run automatic segmentation on images logging predictions as W&B Tables.

## Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
