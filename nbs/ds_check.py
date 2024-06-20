import os, h5py, sys, argparse
import numpy as np


if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--ds", type = str, default = "x15x15") 
        args = parser.parse_args()

        path = "/mnt/sumiya/data/mocat-ml/data/"
        ds = args.ds
        print(ds)

        try:
            data = np.array(h5py.File(f'{path}/TLE_density_all_{ds}.mat', 'r')['comb_Am_rp'])
            corruption = []
            for i, sim in enumerate(data):
                    if np.count_nonzero(sim[-1]) == 0:
                            corruption.append(i)

            print(ds, corruption)

        except :
            print("no such file ", ds)