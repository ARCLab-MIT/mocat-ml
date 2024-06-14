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

def count_nonzeros_over_time(data, stride):
    num_timesteps = data.shape[0]
    zeros_per_timestep = []
    for t in range(0, num_timesteps, stride):
        zeros_per_timestep.append(np.count_nonzero(data[t]))
    return zeros_per_timestep


def plot_nonzeros(ds_name, key, data, stride=8):
    plt.figure(figsize=(8, 6))
    count = count_nonzeros_over_time(data, stride)
    print(len(count))
    plt.plot(np.linspace(0, data.shape[0]-1, len(count)), count)
    plt.xlabel("Timestep")
    plt.ylabel("Number of Non-Zeros")
    plt.title("Number of Non-Zeros Over Time")
    plt.grid(True)
    plt.savefig(f"{ds_name}_{key}_num_of_nonzeros")
    plt.show()
    plt.close()


if __name__ == "__main__":

    keys = ["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]
    for xPop in range(5, 4, -1):
        for xLaunch in range(15, 14, -1):
            for key in keys:
                if xPop + xLaunch == 0: continue

                ds_name = f'x{xPop}x{xLaunch}'
                ds_name = "x8x14"

                # if f'{ds_name}_{key}_num_of_nonzeros.png' in os.listdir(f'plots/{key}/'):
                    # continue

                path = f"/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/corrupted_combined_ds/{ds_name}/TLE_density_all.mat"
                path = f"/home/gridsan/ssarangerel/mocatMC/orbitalrisk_MC/supercloud_runs/combined_ds/{ds_name}/TLE_density_all.mat"

                data = np.array(h5py.File(path, 'r')[key])
                plot_nonzeros(ds_name, key, data[30], 1)

            # print(f"Done x{xPop}x{xLaunch}")
