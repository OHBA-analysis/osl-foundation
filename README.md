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
If you are using the BMRC cluster or hbaws then use the `bmrm.yml` or `hbaws.yml` environment file instead.

## Usage

See the [examples](https://github.com/OHBA-analysis/osl-foundation/tree/main/examples) directory.
