#!/bin/bash

source /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/etc/profile.d/conda.sh
conda activate conda-root-py

wandb online
python data_gen.py
