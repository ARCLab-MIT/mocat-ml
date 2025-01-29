import os, h5py, sys, argparse
import numpy as np

from tqdm import tqdm 

if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--ds", type = str, default = "x15x15") 
        args = parser.parse_args()

        ds = args.ds
        path = f"/home/gridsan/ssarangerel/arclab_shared/mocatml_ds/{ds}"
        # print(ds)

        # try:
        #     data = np.array(h5py.File(f'{path}/TLE_density_all.mat', 'r')['comb_Am_rp'])
        #     corruption = []
        #     for i, sim in enumerate(data):
        #             if np.count_nonzero(sim[-1]) == 0:
        #                     corruption.append(i)

        #     print(ds, corruption)

        # except :
        #     print("no such file ", ds)

        path  = "/home/gridsan/ssarangerel/mocatMC/orbitalrisk_MC/supercloud_runs/combined_ds/"
        
        KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]
        for key in KEYS:
            print(f"Checking key: {key}")

            corrupted = []
            for ds_name in os.listdir(path):
                try:
                    data = np.array(h5py.File(f'/home/gridsan/ssarangerel/arclab_shared/mocatml_ds/{ds_name}/TLE_density_all.mat', 'r')[key])
                    for i, sim in enumerate(data):
                            if np.count_nonzero(sim[-1]) == 0:
                                    corrupted.append(ds_name)
                                    break 

                except :
                    print("no such file or trouble opening", ds_name)
            
            print(key, " Corrupted ", corrupted)

        # Checking key: comb_Am_inc
        # comb_Am_inc  Corrupted  []
        # Checking key: comb_Am_ra
        # comb_Am_ra  Corrupted  []

        # for ds in corrupted:
            # if ds not in os.listdir("/home/gridsan/ssarangerel/mocatMC/orbitalrisk_MC/supercloud_runs/combined_ds/"):
                # print('not addressed', ds)
                