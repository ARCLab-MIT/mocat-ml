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

# Default values for command line arguments
num_runs=1
wandb_group="default_group"

# Function to display help message
show_help() {
cat << EOF
Usage: ${0##*/} [-h] [-r NUM_RUNS] [-g WAND_GROUP]
Run a batch of machine learning experiments with different configurations.

    -h          display this help and exit
    -r NUM_RUNS set the number of runs for each configuration (default: 1)
    -g WANDB_GROUP set the wandb group name (default: "default_group")
EOF
}

# Process command line arguments
while getopts ":hr:g:" opt; do
  case $opt in
    h) 
       show_help
       exit 0
       ;;
    r) num_runs="$OPTARG"
       ;;
    g) wandb_group="$OPTARG"
       ;;
    \?) echo "Invalid option: -$OPTARG" >&2
       show_help
       exit 1
       ;;
  esac
done

# Change to the specified directory
cd ~/Github/mocat-ml/nbs

# Define an array with the desired values
values=(1 2 4 8 16)

# Loop over the values
for val in "${values[@]}"
do
    # Inner loop for the number of runs
    for (( run=1; run<=num_runs; run++ ))
    do
        echo "Running lookback/horizon: ${val}, Run: ${run}"

        # Run the papermill command with the current value for lookback and horizon
        papermill convgru.ipynb pm_tmp_${val}_run${run}.ipynb \
            -p config.device cuda \
            -p config.n_epoch 20 \
            -p config.wandb.enabled True \
            -p config.wandb.log_learner True \
            -p config.wandb.group ${wandb_group} \
            -p config.lookback ${val} \
            -p config.horizon ${val} \
            -p config.sel_steps 100 \
            -p config.stride 2 \
            -p config.bs 32 \
            -p config.save_learner False \
            --prepare-only

        # Run the notebook to python script conversion
        nb2py --nb ./pm_tmp_${val}_run${run}.ipynb --run

        # Remove the temporary notebook
        rm pm_tmp_${val}_run${run}*
    done
done
