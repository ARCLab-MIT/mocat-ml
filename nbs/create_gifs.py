import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_and_save_gif(path, ds_name, stride=8):

    with h5py.File(f'{path}{ds_name}/TLE_density_all.mat', 'r') as mat:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        im = ax.imshow(mat['comb_Am_rp'][0,0,:,:], extent=[np.min(mat['rp_disc']),
                                                        np.max(mat['rp_disc']),
                                                        np.min(mat['Am_disc']),
                                                        np.max(mat['Am_disc'])], aspect='auto')

        plt.xlabel('rp [km]')
        plt.ylabel('Am [m2/kg]')

        def animate(i, im, fig):
            fig.suptitle(f'Epoch {i}')
            im.set_array(mat['comb_Am_rp'][0,stride*i,:, :])  # Update data
            return im,

        ani = animation.FuncAnimation(fig, animate, frames=mat['comb_Am_rp'].shape[1]//stride, fargs=(im, fig))

        # Save the animation as a gif file
        ani.save(f'gifs/{ds_name}.gif', writer='imagemagick', fps=20)
        plt.show()
        plt.close()


if __name__ == "__main__":
    path = "/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/"

    for xPop in range(16):
        for xLaunch in range(16):
            if xPop + xLaunch == 0: continue

            make_and_save_gif(path, f'x{xPop}x{xLaunch}')
            print(f'Done x{xPop}x{xLaunch}')
