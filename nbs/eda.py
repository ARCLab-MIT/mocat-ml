import h5py

name = "TLE_density_all_x9x11.mat"
data = h5py.File(f'/mnt/data/sumiya/mocat-ml/data/{name}', 'r')
print(data.keys())

print(name)
for key in data.keys():
    try:
        print(key, data[key].shape)
    except:
        # print(key, type(data[key]))
        pass
