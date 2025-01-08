import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 

def make_and_save_gif(ds, stride=2):
    with h5py.File(ds, 'r') as mat:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(mat['comb_Am_rp'][0,:,:], extent=[np.min(mat['rp_disc']),
                                                        np.max(mat['rp_disc']),
                                                        np.min(mat['Am_disc']),
                                                        np.max(mat['Am_disc'])], aspect='auto')

        plt.xlabel('rp [km]')
        plt.ylabel('Am [m2/kg]')

        def animate(i, im, fig):
            fig.suptitle(f'Epoch {i}')
            im.set_array(mat['comb_Am_rp'][stride*i,:, :])  # Update data
            return im,

        ani = FuncAnimation(fig, animate, frames=mat['comb_Am_rp'].shape[0]//stride, fargs=(im, fig))

        # Save the animation as a gif file
        ani.save('16x16_data.gif', writer='imagemagick', fps=20)
        plt.show()
        plt.close()


if __name__ == "__main__":
    ds = "/mnt/data/sumiya/mocat-ml/4d_16/x10x10/TLE_density_0.mat"
    make_and_save_gif(ds)
