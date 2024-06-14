import os, h5py, sys
import numpy as np


print("Checking datasets", file=sys.stdout)

ds = []
for i in range(5, 4, -1):
    for j in range(15, 14, -1):
        if i+j==0: continue
        
        path = "/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/"
        ds_name = f'x{i}x{j}'
        print(ds_name)

        data = np.array(h5py.File(f'{path}{ds_name}/TLE_density_all.mat', 'r')['comb_Am_rp'])
        corruption = []
        for i, sim in enumerate(data):
            if np.count_nonzero(sim[-1]) == 0:
                corruption.append(i)
                # print('corrupted at ', ds_name, file=sys.stdout)
        print(corruption)
        # print('Done ', ds_name, file=sys.stdout)

print(' '.join(ds), file=sys.stdout)