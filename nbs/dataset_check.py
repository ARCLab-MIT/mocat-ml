import os, h5py, sys
import numpy as np


print("Checking datasets", file=sys.stdout)

ds = []
for i in range(15, 14, -1):
    for j in range(15, 14, -1):
        if i+j==0: continue
        
        path = "/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/corrupted_combined_ds/"
        ds_name = f'x{i}x{j}'
        
        data = np.array(h5py.File(f'{path}{ds_name}/TLE_density_all.mat', 'r')['comb_Am_rp'])
        corruption = []
        for i, sim in enumerate(data):
            if np.sum(sim[-1]) == 0:
                ds.append(ds_name)  
                corruption.append(i)
                # print('corrupted at ', ds_name, file=sys.stdout)
        print(corruption)
        # print('Done ', ds_name, file=sys.stdout)

print(' '.join(ds), file=sys.stdout)