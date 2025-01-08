from matplotlib import pyplot as plt    
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 
import torch
import torch.nn as nn

def unnormalize_zero_to_one(x, max_val, min_val):
    return (x * (max_val - min_val)) + min_val

def unnormalize_neg_one_to_one(x, max_val, min_val):
    return (x+1) / 2 * (max_val - min_val) + min_val

def exists(x):
    return x is not None

def default(val, d):
    """
    Returns val if it exists, otherwise calls and returns d if it is callable, or returns d if it is not callable.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def plot(lkb, pred, target, save_to, vmin=None, vmax=None):
    n, figsize = 5, (8, 6)
    fig, axs = plt.subplots(nrows = 1, ncols=n, figsize=(figsize[0]*n, figsize[1]), squeeze=False)

    lkb, pred, target = lkb.detach().cpu(), pred.detach().cpu(), target.detach().cpu()
    axs[0, 0].imshow(lkb, aspect = 'auto', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Input')
    axs[0, 1].imshow(pred, aspect = 'auto', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Prediction')
    im = axs[0, 2].imshow(target, aspect = 'auto', vmin=vmin, vmax=vmax)
    axs[0, 2].set_title('Target')
    fig.colorbar(im, ax=axs[0, :3])

    loss_im = axs[0, 4].imshow(np.abs(pred-target), cmap = 'Blues', aspect = 'auto', vmin=vmin, vmax=vmax)
    axs[0, 4].set_title(f'Absolute diff mean: {round(torch.mean(torch.abs(pred-target)).item(), 2)} | max: {round(torch.max(torch.abs(pred-target)).item(), 2)}')
    fig.colorbar(loss_im, ax=axs[0, 4])

    axs[0, 5].hist(np.abs(pred-target).flatten(), bins=200) 
    axs[0, 5].set_title('Distribution of Absolute Difference Values')

    fig.savefig(save_to)    
    plt.close()


def make_and_save_gif(preds, savename, vmin, vmax, stride=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(preds[0,:,:], aspect='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)

    def animate(i, im, fig):
        fig.suptitle(f'Forecast iteration {i} target|prediction')
        im.set_array(preds[stride*i,:, :])  # Update data
        return im,

    ani = FuncAnimation(fig, animate, frames=preds.shape[0]//stride, fargs=(im, fig))
    
    # Save the animation as a gif file
    ani.save(savename, writer='imagemagick', fps=20)
    plt.close()


def plot_losses(train_losses, val_losses, save_to):
    plt.figure(figsize=(8, 5))  
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(save_to)
    plt.close()