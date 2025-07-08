import numpy as np
import torch, math
from matplotlib import pyplot as plt    
from matplotlib.animation import FuncAnimation

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

    loss_im = axs[0, 3].imshow(np.abs(pred-target), cmap = 'Blues', aspect = 'auto', vmin=vmin, vmax=vmax)
    axs[0, 3].set_title(f'Absolute diff mean: {round(torch.mean(torch.abs(pred-target)).item(), 2)} | max: {round(torch.max(torch.abs(pred-target)).item(), 2)}')
    fig.colorbar(loss_im, ax=axs[0, 3])

    axs[0, 4].hist(np.abs(pred-target).flatten(), bins=200) 
    axs[0, 4].set_title('Distribution of Absolute Difference Values')

    fig.savefig(save_to)    
    plt.close()


def make_and_save_gif(target, preds, config, savename, stride=10):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))  # Create a figure with three subplots
    vmin, vmax = min([torch.min(target).item(), torch.min(preds).item()]), max([torch.max(target).item(), torch.max(preds).item()])
    abs_diff = torch.abs(target-preds)  
    n_iter = 2436//(config.lookback  + config.gap) - math.ceil(config.lookback/(config.lookback + config.gap))
    
    # Create images for the first and second distributions
    im1 = ax1.imshow(target[0, :, :], aspect='auto', vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(preds[0, :, :], aspect='auto', vmin=vmin, vmax=vmax)
    im3 = ax3.imshow(abs_diff[0, :, :], aspect='auto', cmap='viridis', interpolation='nearest', vmin=torch.min(abs_diff).item(), vmax=torch.max(abs_diff).item())    

    # Add colorbars
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    ax1.set_title("MOCAT-MC")
    ax2.set_title("MOCAT-ML")
    ax3.set_title("Absolute Difference")
    ax4.set_title("Model Configuration")

    # Display model configuration in the third subplot
    model_config = {key:config[key] for key in ['launch_rate', 'init_pop', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'key', 'loss', 'sample', 'log', 'average', 'metric']}
    config_text =  "\n\n" + "\n".join([f"{key}: {value}" for key, value in model_config.items()])
    ax4.text(0.05, 0.3, config_text, fontsize=8) 
    ax4.axis('off') 

    def animate(i):
        im1.set_array(target[i * stride, :, :])
        im2.set_array(preds[i * stride, :, :])
        im3.set_array(abs_diff[i * stride, :, :])

        # Update the title with the current iteration and year
        fig.suptitle(f'Forecast iteration {round(i * stride / 2436 * n_iter)} year: {round(i * stride / 2436 * 100)}')
        return im1, im2, im3, fig

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=target.shape[0] // stride, interval=20, blit=True)

    # Save the animation as a GIF
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