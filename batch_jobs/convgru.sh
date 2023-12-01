#!/bin/bash
#SBATCH --job-name=mocatml_convgru
#SBATCH --mail-type=END
#SBATCH --mail-user=victorrf@mit.edu
#SBATCH --output=res.txt
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading the required module
source /etc/profile
module load anaconda/2023a

# Run the script
cd ~/Github/mocat-ml/nbs
papermill convgru.ipynb pm_tmp.ipynb \
	-p config.device cuda \
	-p config.n_epoch 20 \
	-p config.wandb.enabled True \
	-p config.wandb.log_learner False \
	-p config.lookback 4 \
	-p config.horizon 4 \
	-p config.gap -3 \
	-p config.stride 4 \
	-p config.bs 32 \
	-p config.save_learner True \
	-p config.lr_max 0.001 \
	--prepare-only

nb2py --nb ./pm_tmp.ipynb --run
#rm pm_tmp*
