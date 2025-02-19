import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from tsai.imports import my_setup
from tsai.utils import yaml2dict
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, torch, random, datetime
import numpy as np

from loss_functions import *
from utils import *
from diffusion_utils import make_and_save_gif

import matplotlib.pyplot as plt

my_setup()


def make_gif(preds, stride=10):
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(preds[0, :, :], aspect='auto', vmin=preds.min(), vmax=preds.max())
    fig.colorbar(im, ax=ax)

    # Define the update function
    def update(frame):
        """Update the heatmap and annotations for each animation frame."""
        # Extract the data for the current frame
        im.set_array(preds[frame*stride, :, :])
        return im, 

    # Create the figure and axis
    ani = FuncAnimation(fig, update, frames=preds.shape[0]//stride, interval=50, blit=True) 
    ani.save("gif.gif", writer='imagemagick', fps=20) 
    plt.show()


def get_ds(ds_name, config):  
    data_path = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam"
    path = f'{data_path}/TLE_density_all_{ds_name}.mat'
    mat = h5py.File(path, 'r')
    data = np.array(mat[config.key])[:, :config.sel_steps]
    _, timesteps, w, h, c = data.shape
    data = data.sum(axis=-1)

    if config.average:
        data = data.reshape(5, -1, timesteps, w, h).mean(axis=1)

    if config.log:
        data = np.log(data + 1)

    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw


def get_dls(ds_list, config):
    train_dls, valid_dls, splits, Xs = [], [], [], []
    for ds_name in ds_list:
        X, X_sw = get_ds(ds_name, config)
        split = RandomSplitter()(X) #valid_pct=0.2,
        ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
        
        samples_per_simulation = X_sw.shape[0]//(X.shape[0])
        train_idxs = calculate_sample_idxs(split[0], samples_per_simulation)
        valid_idxs = calculate_sample_idxs(split[1], samples_per_simulation)

        print(ds_name, X.shape, X_sw.shape)

        train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
        valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))

        train_dls.append(train_tl)
        valid_dls.append(valid_tl)
        splits.append(split)
        Xs.append(X)

    train = np.concatenate([X[split[0]] for X, split in zip(Xs, splits)], axis=0)   
    mocat_stats = (np.mean(train), np.std(train))

    train, valid = ConcatDataset(train_dls), ConcatDataset(valid_dls)
    dls = DataLoaders.from_dsets(train, valid, bs=config.bs, device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.norm else None,
                        num_workers=config.num_workers)

    return dls, splits, Xs



def main(ds_list, config):
    # data setup
    dls, splits, Xs  = get_dls(ds_list, config)
    loss_func, metrics = get_loss_function(config), get_metrics(config)


    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    learn = Learner(dls, model, loss_func=loss_func, cbs=[], metrics=metrics)
    learn.splits = splits # This is needed for the evaluation notebook
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()


    # Training
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    print("Training DONE!")

    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f"forecasts/{date}")
    os.makedirs(f"smapes/{date}")
    
    # Evalution
    model.eval()
    for X, split, ds in zip(Xs, splits, ds_list):
        ind = random.choice(split[1])
        val = torch.tensor(X[ind])

        initial_seq = torch.tensor(val[:config.lookback], device=default_device()).float()
        inp = tuple(i.reshape(1,1,32,32) for i  in initial_seq)

        n_iter = 2436//config.lookback
        preds = [initial_seq]
        for _ in range(n_iter):
            p = model(inp)
            preds.append(torch.squeeze(torch.vstack(p), dim=1))
            inp = p

        preds = torch.vstack(preds)[:2436].cpu()
        smapes = [newSMAPE(config.log)(i, j).item() for i,j in zip(preds[config.lookback:], val[config.lookback:])]
        
        # SMAPE
        fig = plt.figure()
        fig.clf()
        plt.plot(smapes, label=f'SMAPE ds: {ds}')
        plt.legend()
        plt.savefig(f'smapes/{date}/smape_{ds}.png')
        plt.show()

        if config.log:
            val, preds = torch.exp(val)-1, torch.exp(preds)-1

        # Forecasts of N
        make_and_save_gif(val, preds.detach(), config, f"forecasts/{date}/{ds}.gif")


        # Forecasts of log-N
        make_and_save_gif(torch.log(val+1), torch.log(preds.detach()+1), config, f"forecasts/{date}/{ds}_logN.gif")


    # Save model
    if config.save_model:
        torch.save(learn.model.state_dict(), f"pretrained_model/model_ip_{config.init_pop}_lr_{config.launch_rate}_pmd_{config.pmd}_cam_{config.cam}.pth")


# TODO script to produce a grid of smape values across ip/lr/pmd/cam

if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")


    # Data settings 
    parser.add_argument("--init_pop", nargs="+", type=int, default = [1, 1])   # range of initial populations to use for training
    parser.add_argument("--launch_rate", nargs="+", type=int, default = [2, 2])  # range of launch rates to use for training 
    parser.add_argument("--pmd", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--cam", nargs="+", type=float, default=[0.90, 0.95, 0.96, 0.97, 0.98, 0.99])
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--lookback", type = int, default = 50)
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 100) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--n_epoch", type = int, default = 30)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--log", type = int, default = 0)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    parser.add_argument("--norm", type = int, default = 1) # whether or not to normalize (Z-norm) during training
    parser.add_argument("--save_model", type = int, default = 0) # whether or not to save the model


    # Set defaults 
    args = parser.parse_args()
    args.init_pop = [i for i in range(args.init_pop[0], args.init_pop[1]+1)]    
    args.launch_rate = [i for i in range(args.launch_rate[0], args.launch_rate[1]+1)]   
    args.ds_list = [f'x{init_pop}x{launch_rate}x{pmd:.2f}x{cam:.2f}' for init_pop in args.init_pop for launch_rate in args.launch_rate for pmd in args.pmd for cam in args.cam] # Default datasets to train on
    arg_dict = vars(args)


    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)

    config.partial_loss = [0] if args.partial_loss == 1 else None
    keys = ['ds_list', 'launch_rate', 'init_pop', 'pmd', 'cam', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'log', 'average', 'norm', 'save_model']
    for key in keys:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    main(args.ds_list, config)    
