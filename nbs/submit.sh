#!/bin/bash

#SBATCH -c 40
#SBATCH --gres=gpu:volta:1
#SBATCH -o 16x16_output.sh.log-%j-%a

# Loading the required module
source /etc/profile
module load anaconda/2023b
source activate mocat-ml

# Run the script
python model_train.py --loss mae --horizon 16 --lookback 16