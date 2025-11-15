# OHBA Software Library: Foundation Model Toolbox

This repository contains a tokeniser and foundation model (MEG-GPT) for parcellated MEG data.

Preprint: https://arxiv.org/abs/2510.18080.

## Installation

We recommend using Mamba (`mamba`) to install osl-foundation. First install Miniforge3 (`conda`):
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```
Then install `mamba` with:
```
conda install -n base -c conda-forge mamba
```

osl-foundation can be installed with:
```
git clone https://github.com/OHBA-analysis/osl-foundation.git
cd osl-foundation
mamba env create -f envs/oslf.yml
conda activate oslf
pip install -e .
```
Note, MEG-GPT requires TensorFlow 2.11.

### Oxford-specific computers

#### Biomedical Research Computing (BMRC) Cluster

`conda`/`mamba` are available as a software module:
```
module load Miniforge3
```
and osl-foundation can be installed with:
```
git clone https://github.com/OHBA-analysis/osl-foundation.git
cd osl-foundation
mamba env create -f envs/bmrc.yml
conda activate oslf
pip install -e .
```
Note, the following CUDA module needs to be loaded on BMRC to use TensorFlow:
```
module load cuDNN/8.4.1.50-CUDA-11.7.0
```

#### hbaws

First install `conda`/`mamba` using the instructions above, then install osl-foundation with:
```
git clone https://github.com/OHBA-analysis/osl-foundation.git
cd osl-foundation
mamba env create -f envs/hbaws.yml
conda activate oslf
pip install -e .
```

## Usage

See the [examples](https://github.com/OHBA-analysis/osl-foundation/tree/main/examples) directory.
