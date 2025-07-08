import numpy as np
import h5py
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--init_pop", nargs="+", type=int, default = [1, 2])   # range of initial populations to use for training
    parser.add_argument("--launch_rate", nargs="+", type=int, default = [0, 2])  # range of launch rates to use for training 
    parser.add_argument("--pmd", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--cam", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--lookback", type = int, default = 50) 
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    args = parser.parse_args()

    all_data = []  # List to hold all time series data
    for ip in range(args.init_pop[0], args.init_pop[1] + 1):
        for lr in range(args.launch_rate[0], args.launch_rate[1] + 1):
            for pmd in tqdm(args.pmd):
                for cam in args.cam:
                    path = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam/"
                    ds_name = f"x{ip}x{lr}x{pmd:.2f}x{cam:.2f}"
                    file = f"TLE_density_all_{ds_name}.mat"

                    with h5py.File(path + file, 'r') as f:
                        ds = np.array(f[args.key])
                        if args.average:
                            ds = ds.mean(axis=0)

                        all_data.append(ds[:args.lookback])

    # Convert to NumPy array *after* gathering all data
    all_data = np.array(all_data)

    # Save data with compression and chunking
    with h5py.File(f"../example_data/{args.key}_ip_[{args.init_pop[0]},{args.init_pop[1]}]_lr_[{args.launch_rate[0]},{args.launch_rate[1]}]_pmd_{args.pmd}_cam_{args.cam}.hdf5", "w") as f:
        dset = f.create_dataset("time_series", data=all_data, compression="gzip", chunks=all_data.shape)  # Chunking added

        # Store the dataset names for easy access later.
        ds_names = [f"x{ip}x{lr}x{pmd:.2f}x{cam:.2f}" for ip in range(args.init_pop[0], args.init_pop[1] + 1) for lr in range(args.launch_rate[0], args.launch_rate[1] + 1) for pmd in args.pmd for cam in args.cam]
        f.create_dataset("dataset_names", data=np.array(ds_names, dtype=h5py.string_dtype()))  # Save dataset names
