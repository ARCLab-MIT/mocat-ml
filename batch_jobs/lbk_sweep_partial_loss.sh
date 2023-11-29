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
# Define an array with the desired values
values=(1 2 4 8 16)

# Loop over the values
for val in "${values[@]}"
do
    # Run the papermill command with the current value for lookback and horizon
    papermill convgru.ipynb pm_tmp_${val}.ipynb \
	-p config.device cuda \
        -p config.n_epoch 20 \
        -p config.wandb.enabled True \
        -p config.wandb.log_learner False \
	-p config.wandb.group lbk_sweep_partial_loss \
        -p config.lookback ${val} \
        -p config.horizon ${val} \
	-p config.sel_steps 100 \
        -p config.stride 2 \
        -p config.bs 32 \
	-p config.save_learner False \
        --prepare-only

    # Run the notebook to python script conversion
    nb2py --nb ./pm_tmp_${val}.ipynb --run

    # Remove the temporary notebook
    rm pm_tmp_${val}*
done
