# OHBA Software Library: Foundation Model Toolbox

This repository contains a tokeniser and foundation model (MEG-GPT) for parcellated MEG data.

Preprint: https://arxiv.org/abs/2510.18080.

## Installation

```
git clone https://github.com/OHBA-analysis/osl-foundation.git
cd osl-foundation
conda env create -f envs/oslf.yml
conda activate oslf
pip install -e .
```
Note, MEG-GPT requires TensorFlow 2.11.

### Oxford-specific computers

If you are using the BMRC cluster, use the `bmrc.yml` environment file instead of `oslf.yml` and load the following CUDA module:
```
module load cuDNN/8.4.1.50-CUDA-11.7.0
```

On hbaws, use the the `hbaws.yml` environment file.

## Usage

See the [examples](https://github.com/OHBA-analysis/osl-foundation/tree/main/examples) directory.
