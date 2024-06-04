import h5py, os
import numpy as np
import matplotlib.pyplot as plt

def print_ds(ds_name):
    path = f"/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/{ds_name}/TLE_density_all.mat"
    data = h5py.File(path, 'r')
    for key in data.keys():
        try:
            print(key, data[key].shape)
        except:
            print(key, type(data[key]))
        pass

def count_nonzeros_over_time(data, stride=8):
    num_timesteps = data.shape[0]
    zeros_per_timestep = []
    for t in range(0, num_timesteps, stride):
        zeros_per_timestep.append(np.count_nonzero(data[t]))
    return zeros_per_timestep


def plot_nonzeros(ds_name, key, data):
    plt.figure(figsize=(8, 6))
    plt.plot(range(data.shape[0]), count_nonzeros_over_time(data))
    plt.xlabel("Timestep")
    plt.ylabel("Number of Non-Zeros")
    plt.title("Number of Non-Zeros Over Time")
    plt.grid(True)
    path = f'plots/{key}'
    if not os.path.exists(path):
        os.makedirs(path)
        
    plt.savefig(f"plots/{key}/{ds_name}_{key}_num_of_nonzeros")
    plt.show()
    



if __name__ == "__main__":


    keys = ["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]
    for xPop in range(16):
        for xLaunch in range(16):
            for key in keys:
                if xPop + xLaunch == 0: continue

                ds_name = f'x{xPop}x{xLaunch}'
                path = f"/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/{ds_name}/TLE_density_all.mat"
                data = np.array(h5py.File(path, 'r')[key])
                plot_nonzeros(ds_name, key, data[0])
                