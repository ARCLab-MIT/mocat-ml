import sys
sys.path.append('..')
from tsai.utils import yaml2dict
from utils import *
import argparse, torch, math

from fastcore.all import *
from fastai.vision.all import *
from torchvision import transforms


def forecast(config):
    # Load pretrained model 
    model = load_model(config)
    
    # Load data
    initial_data = np.load(config.data_path)[f'x{config.init_pop}x{config.launch_rate}'][:50]
    initial_data = torch.tensor(initial_data, device= default_device()).float()
    transform = transforms.Compose([
        transforms.Resize((32, 32))
    ])
    initial_data = transform(initial_data).reshape(50, 1, 1, 32, 32)
    inp = tuple(i for i  in initial_data)
    n_iter = 2436//50
    predictions = [initial_data.squeeze()]
    for _ in range(n_iter):
        p = model(inp)
        p = torch.squeeze(torch.vstack(p), dim=1)
        predictions.append(p)
        inp = tuple(i.reshape(1, 1, 32, 32) for i in p)

    # Save the predictions
    predictions = torch.vstack(predictions) 
    return predictions


def load_model(config): 
    # model setup
    cfg = AttrDict(yaml2dict(f'./config/convgru/convgru.yaml', attrdict=True))
    model = StackUnstack(SimpleModel(**cfg)).to(default_device())
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    return model


def make_gif(preds, stride=10):
    preds = preds[:2436, :, :]  
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Create a figure with three subplots
    n_iter = 2436//50
    vmin, vmax = preds.min(), preds.max()
    
    im = ax.imshow(preds[0, :, :], aspect='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title("MOCAT-ML forecast")

    def animate(i):
        im.set_array(preds[i * stride, :, :])

        # Update the title with the current iteration and year
        fig.suptitle(f'Forecast iteration {round(i * stride / 2436 * n_iter)} year: {round(i * stride / 2436 * 100)}')
        return im, fig

    # Create the animation
    return FuncAnimation(fig, animate, frames=preds.shape[0] // stride, interval=20, blit=True)



if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "100 year forecasting")

    # Settings 
    parser.add_argument("--launch_rate", type=int, default = 2)  
    parser.add_argument("--init_pop", type=int, default = 2)   
    parser.add_argument("--model_path", type = str, default = "../pretrained_model/model_1_12_1_12.pth")
    parser.add_argument("--sample", type = int, default = 1) # whether or not to sample to 32x32
    parser.add_argument("--data_path", type = str, default = "../example_data/data.npz")
    args = parser.parse_args()

    # Inference 
    preds = forecast(args)

    # Visualization
    gif = make_gif(preds.cpu().detach())
    gif.save('forecast.gif') 
