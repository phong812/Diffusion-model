<div align="center">

# DDPM

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

In this repository, I implemented two basic diffusion models (DDPM & DDIM). Both models were trained on the MNIST, FashionMNIST, and CIFAR10 datasets. The code for this implementation was referenced from [here](https://github.com/awjuliani/pytorch-diffusion/tree/master).

Supports MNIST, Fashion-MNIST and CIFAR datasets.


## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU (default)
python src/train.py
```

## Generated Images
### FashionMNIST
[![image](https://github.com/user-attachments/assets/cc200323-1860-4428-a0ae-92870dd79c00)](https://github.com/phong812/Diffusion-model/blob/6dd0731236e1aea71919f13f386aeddab0b2bc0d/pred.gif)

