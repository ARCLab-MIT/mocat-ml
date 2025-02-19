import sys
sys.path.append('..')
from tsai.utils import yaml2dict
from utils import *
import argparse, torch, h5py

from fastcore.all import *
from fastai.vision.all import *

import matplotlib.colors as mcolors



def forecast(config):
    # Load pretrained model 
    model = load_model(config)
    
    # Load data
    ds_name = f'x{config.init_pop}x{config.launch_rate}x{config.pmd:.2f}x{config.cam:.2f}'
    with h5py.File(config.data_path, 'r') as f:
        time_series_data = f['time_series'][:]  
        dataset_names = f['dataset_names'][:].astype(str).tolist() 
        ind = dataset_names.index(ds_name)
        initial_data = time_series_data[ind][:50].sum(axis=-1)

    # Forecasting
    initial_data = torch.log(torch.tensor(initial_data, device= default_device()).float()+1)
    inp = tuple(i.reshape(1,1,32,32) for i  in initial_data)

    n_iter = 2436//50
    predictions = [initial_data.squeeze()]
    for _ in range(n_iter):
        p = model(inp)
        predictions.append(torch.squeeze(torch.vstack(p), dim=1))
        inp = p

    # Save the predictions
    predictions = torch.vstack(predictions).cpu().detach() 
    predictions = torch.exp(predictions[:2436])-1
    return predictions



def load_model(config): 
    # Model setup
    cfg = AttrDict(yaml2dict(f'./config/convgru/convgru.yaml', attrdict=True))
    model = StackUnstack(SimpleModel(**cfg)).to(default_device())
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    return model



def make_gif(preds, stride=10):
    preds += 1 # To avoid error when plotting with log scale

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Create a figure with three subplots
    n_iter = 2436//50
    vmin, vmax = preds.min(), preds.max()

    extent = [6578.137, 8378.136999999999, 9.999999999999999e-06, 10.0]
    im = ax.imshow(preds[0, :, :], aspect='auto', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), extent = extent)

    fig.colorbar(im, ax=ax)
    ax.set_title("MOCAT-ML forecast")

    plt.xlabel('rp [km]')
    plt.ylabel('Am [m2/kg]')
    
    def animate(i):
        im.set_array(preds[i*stride, :, :])

        # Update the title with the current iteration and year
        fig.suptitle(f'Forecast iteration {round(i*stride / 2436*n_iter)} year: {round(i*stride / 2436*100)}')
        return im, fig

    # Create the animation
    return FuncAnimation(fig, animate, frames=preds.shape[0] // stride, interval=20, blit=True)



if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "100 year forecasting")

    # Settings 
    parser.add_argument("--init_pop", type = int, default = 1)   
    parser.add_argument("--launch_rate", type = int, default = 2)   
    parser.add_argument("--pmd", type = float, default = 0.90)
    parser.add_argument("--cam", type = float, default = 0.90)
    parser.add_argument("--model_path", type = str, default = "../pretrained_model/model_ip_[1, 2]_lr_[1, 2]_pmd_[0.9, 0.95, 0.96, 0.97, 0.98, 0.99]_cam_[0.9, 0.95, 0.96, 0.97, 0.98, 0.99].pth")
    parser.add_argument("--data_path", type = str, default = "../example_data/comb_Am_rp_ip_[1,2]_lr_[0,2]_pmd_[0.9, 0.95, 0.96, 0.97, 0.98, 0.99]_cam_[0.9, 0.95, 0.96, 0.97, 0.98, 0.99].hdf5")
    args = parser.parse_args()

    # Inference 
    preds = forecast(args)

    # Visualization 
    gif = make_gif(preds)
    gif.save('forecast.gif') 
