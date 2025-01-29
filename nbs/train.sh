#!/bin/bash 

#SBATCH --job-name=MOCAT-ML-training-ip-[1,10]-lr-[1,10]
#SBATCH -o MOCAT-ML.logs-%j
#SBATCH --output=logs/MOCAT-ML-training-ip-[1,10]-lr-[1,10]--%A_%a.out
#SBATCH --error=logs/MOCAT-ML-training-ip-[1,10]-lr-[1,10]--%A_%a.err 
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

cd /home/gridsan/ssarangerel/mocat-ml/nbs
python -u model_train.py --launch_rate 1 10 --init_pop 1 10 >> logs/output.txt
