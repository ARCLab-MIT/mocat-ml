import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import pandas as pd

KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]
FSIZE, LBL_SIZE, TITLE_SIZE=14,16,18
SOLAR_CYCLE = 12 # in years



def add_guidance(data, params):
    """
    Appends paramaters and time embeddings as additional channels to the input tensor.
    """

    params = params[1:] # exclude initial population

    num_sim, timesteps, c, w, h = data.shape
    num_params = len(params)

    timestamps = np.linspace(0, 100, timesteps).astype(np.float32)
    # time embedding relative to solar cycle
    timestamps = np.mod(timestamps, SOLAR_CYCLE)

    # Expand and broadcast parameters
    params_tensor = np.array(params).reshape(1, num_params, 1, 1)
    params_tensor = np.broadcast_to(params_tensor, (num_sim, num_params, w, h))
    params_tensor = np.expand_dims(params_tensor, axis=1)
    params_tensor = np.broadcast_to(params_tensor, (num_sim, timesteps, num_params, w, h))

    # Expand and broadcast timestamps
    timestamps_tensor = timestamps.reshape(1, timesteps, 1, 1, 1)
    timestamps_tensor = np.broadcast_to(timestamps_tensor, (num_sim, timesteps, 1, w, h))  
    return np.concatenate([data, params_tensor, timestamps_tensor], axis=2)



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp




def save_training_metrics(learn, metrics, save_folder):
    """
    Saves training metrics such as training and validation loss, and metric (smape)
    """
    history = learn.recorder.values
    metric_names = [m.__name__ for m in metrics]
    columns = ['train_loss', 'valid_loss'] + metric_names
    df = pd.DataFrame(history, columns=columns)

    # Save to CSV
    df.to_csv(f"{save_folder}training_metrics.csv", index=False)    


    # Extract losses and metrics
    epochs = range(1, len(history) + 1)
    train_loss = [row[0] for row in history]
    valid_loss = [row[1] for row in history]
    metric_values = list(zip(*[row[2:] for row in history]))  # Unzips metrics for separate plotting

    # Plot Training & Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o', linestyle='-', color='blue')
    plt.plot(epochs, valid_loss, label="Valid Loss", marker='o', linestyle='-', color='red')
    plt.xlabel("Epochs", fontsize=LBL_SIZE)
    plt.ylabel("MAE Loss", fontsize=LBL_SIZE)
    plt.title("Training & Validation Loss Evolution", fontsize=TITLE_SIZE)
    plt.xticks(fontsize=FSIZE)
    plt.yticks(fontsize=FSIZE)
    plt.legend(fontsize=FSIZE)
    plt.grid()
    plt.savefig(f"{save_folder}train_val_loss.png")

    # Plot All Metrics Together
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        plt.plot(epochs, metric_values[i], label=metric, marker='o', linestyle='--')

    plt.xlabel("Epochs", fontsize=LBL_SIZE)
    plt.ylabel("Metric Value",fontsize=LBL_SIZE)
    plt.title("Metrics Evolution", fontsize=TITLE_SIZE)
    plt.xticks(fontsize=FSIZE)
    plt.yticks(fontsize=FSIZE)
    plt.legend(fontsize=FSIZE)
    plt.grid()
    plt.savefig(f"{save_folder}metrics.png")



def calculate_sample_idxs(simulation_idxs, samples_per_sim):
    indices = []
    for sim in simulation_idxs:
        start_idx = sim * samples_per_sim
        end_idx = start_idx + samples_per_sim
        indices.extend(range(start_idx, end_idx))
    return indices
