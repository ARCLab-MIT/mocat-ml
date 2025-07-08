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
import wandb, json, argparse, os, torch, random, datetime, h5py, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from loss_functions import *
from nbs.utils import add_guidance, get_n_params, save_training_metrics


RANDOM_DATA_PATH = '/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/new_dataset/'
DATA_PATH = "/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/"
VALIDATIOON_DATA_PATH = '/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam/'
KEY, N, NORM = 'comb_Am_rp', 2436, 13
SOLAR_CYCLE = 12 # in years
FSIZE, LBL_SIZE, TITLE_SIZE = 14, 16, 18


def get_data(ds_name):
    path = f'{DATA_PATH}{ds_name}/TLE_density_all.mat'
    mat = h5py.File(path, 'r')
    data = np.array(mat[KEY]).copy()
    return data



def get_train_val_data(args, ds_name):

    data = get_data(ds_name)
    _, timesteps, w, h = data.shape

    if args.average:
        data = data.reshape(5, -1, timesteps, 1,  w, h).mean(axis=1) # Add channel dimension as well

    if args.log:
        data = np.log(data + 1)

    if args.guidance:
        params = [int(i) for i in ds_name.split('x')[1:]]
        data = add_guidance(data, params)

    return data[:4], data[4:5] # first 4 for training last one for validation



def data_to_data_sw(data):
    data_sw = np.lib.stride_tricks.sliding_window_view(data, args.lookback + args.horizon + args.gap, axis=1)[:,::args.stride,:]
    data_sw = data_sw.transpose(0,1,5,2,3,4)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data_sw



def get_dls(args):
    ip_min, ip_max = args.ip
    lr_min, lr_max = args.lr

    datasets = [f'x{i}x{j}' for i in range(ip_min, ip_max+1) for j in range(lr_min, lr_max+1)]

    train_dls, train_Xs = [], []
    valid_dls, valid_Xs = [], []
    for ds_name in datasets: 
        
        train_X, val_X = get_train_val_data(args, ds_name)
        train_X_sw, val_X_sw  = data_to_data_sw(train_X), data_to_data_sw(val_X)

        train_ds = DensityData(train_X_sw, lbk=args.lookback, h=args.horizon, gap=args.gap)
        samples_per_simulation = train_X_sw.shape[0]//(train_X.shape[0])
        train_tl = TfmdLists(calculate_sample_idxs(range(train_X.shape[0]), samples_per_simulation), DensityTupleTransform(train_ds))
        train_dls.append(train_tl)
        train_Xs.append(train_X)

        print("Loaded training dataset: ", ds_name, " Shapes: ", train_X.shape, train_X_sw.shape)

        val_ds = DensityData(val_X_sw, lbk=args.lookback, h=args.horizon, gap=args.gap)
        samples_per_simulation = val_X_sw.shape[0]//(val_X.shape[0])
        val_tl = TfmdLists(calculate_sample_idxs(range(val_X.shape[0]), samples_per_simulation), DensityTupleTransform(val_ds))
        valid_dls.append(val_tl)
        valid_Xs.append(val_X)
        
        print("Loaded validation dataset: ", ds_name, " Shapes: ", val_X.shape, val_X_sw.shape)


    train = np.concatenate(train_Xs, axis=0)   
    mocat_stats = (np.mean(train), np.std(train))

    train, valid = ConcatDataset(train_dls), ConcatDataset(valid_dls)
    dls = DataLoaders.from_dsets(train, valid, bs=config.bs, device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)],
                        num_workers=config.num_workers)

    return dls



def train(config):

    # data setup
    dls = get_dls(config)
    loss_func, metrics = StackLoss(MAE()), get_metrics_ip_lr(config)

    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    learn = Learner(dls, model, loss_func=loss_func, cbs=[], metrics=metrics)
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()

    # Training
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    print("Training DONE!")

    # # Directories to save results 
    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f"results/{date}")
    os.makedirs(f"models/{date}")

    # Save training history
    save_training_metrics(learn, metrics, save_folder=f'zresults/{date}/')

    # Save config as txtfile
    with open(f"zresults/{date}/config.txt", 'w') as f:
        json.dump(config, f, indent=4)

    # Save model
    if config.save_model:
        torch.save(learn.model.state_dict(), f"results/{date}/model.pth")

        g, h, n = config.guidance, config.horizon, config.n_epoch
        torch.save(learn.model.state_dict(), f"models/{date}/ip_{config.ip}_lr_{config.lr}_guidance_{g}_horizon_{h}_n_epochs_{n}.pth")


    # Evalution
    with torch.no_grad():
        model.eval()

        ip_min, ip_max = config.ip
        lr_min, lr_max = config.lr

        datasets = [f'x{i}x{j}' for i in range(ip_min, ip_max+1) for j in range(lr_min, lr_max+1)]
        
        for ds in datasets:
            val = get_data(ds).mean(axis=0, keepdims=True) 
            val = np.expand_dims(val, axis=2) 
            val = np.log(val+1)
            
            if config.guidance:
                params = [int(i) for i in ds.split('x')[1:]]
                val = add_guidance(val, params)

            val = torch.tensor(val[0]).float()

            # Iterative forecasting
            initial_seq = val[:config.lookback].to(default_device())
            inp = tuple(i.unsqueeze(0) for i in initial_seq)

            preds = [initial_seq[:,:1,:,:].cpu().detach()]
            
            n_iter = N//config.lookback
            for iter in range(1, n_iter+1):
                p = model(inp)

                preds.append(torch.vstack(p).cpu().detach())

                if iter==n_iter:
                    break

                if config.guidance:
                    target = val[iter*config.horizon:(iter+1)*config.horizon].to(default_device())
                    guidance = target[:,1:,:,:] 
                    inp = tuple(torch.cat([p[i], guidance[i:i+1]], axis=1)  for i in range(config.lookback))

                else:
                    inp = p
                
            preds = torch.vstack(preds)[:N]

            if config.guidance:
                val = val[:,:1,:,:]

            plot_save_results(val, preds, config, ds, date, metrics)



def plot_save_results(val, preds, config, ds, date, metrics):

    # Directory to save results
    os.makedirs(f"results/{date}/{ds}")
    years = np.linspace(0, 100, 2436)[config.lookback:]


    # Plotting metrics over time 
    for smape in metrics:
        smapes = [smape.loss_func(i, j).item() for i,j in zip(preds[config.lookback:], val[config.lookback:])]
        fig = plt.figure()
        fig.clf()
        plt.plot(years, smapes, label=f'ds: {ds}')
        plt.xlabel("Years", fontsize=LBL_SIZE)
        plt.title(f"{smape.__name__} over time", fontsize=TITLE_SIZE)
        plt.xticks(fontsize=FSIZE)
        plt.yticks(fontsize=FSIZE)
        plt.legend(fontsize=FSIZE)
        plt.tight_layout()
        plt.savefig(f'zresults/{date}/{ds}/{smape.__name__}.png')
        plt.show()
        plt.close()

    if config.log:
        val, preds = torch.exp(val)-1, torch.exp(preds)-1

    # Plot loss over time 
    maes = [L1LossFlat()(i, j).item() for i, j in zip(preds[config.lookback:], val[config.lookback:])]
    fig = plt.figure()
    fig.clf()
    plt.plot(years, maes, label=f'Losses ds: {ds}')
    plt.title("Mean absolute error over time", fontsize=TITLE_SIZE)
    plt.xlabel("Years", fontsize=LBL_SIZE)
    plt.xticks(fontsize=FSIZE)
    plt.yticks(fontsize=FSIZE)
    plt.legend(fontsize=FSIZE)
    plt.tight_layout()
    plt.savefig(f'results/{date}/{ds}/losses.png')
    plt.show()
    plt.close()


    # Total number of objects over time
    mc, ml = val.sum(dim=(-1,-2))[config.lookback:], preds.sum(dim=(-1,-2))[config.lookback:]
    fig = plt.figure()
    fig.clf()
    plt.plot(years, mc, label=f'MC')
    plt.plot(years, ml, label=f'ML')
    plt.title("Total number of objects over time", fontsize=TITLE_SIZE)
    plt.xlabel("Years", fontsize=LBL_SIZE)
    plt.xticks(fontsize=FSIZE)
    plt.yticks(fontsize=FSIZE)
    plt.legend(fontsize=FSIZE)
    plt.tight_layout()
    plt.savefig(f'results/{date}/{ds}/total_num_objects.png')
    plt.show()
    plt.close()

    val, preds = val.squeeze(), preds.squeeze()
    # Produce gif for the prediction
    make_and_save_gif(val, preds, config, f"results/{date}/{ds}/forecast.gif")




def make_and_save_gif(target, preds, config, savename, stride=10):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))  # Create a figure with three subplots
    preds, target, abs_diff = preds+1, target+1, torch.abs(target-preds)+1 # Adding one to avoid error when putting log scale 
    
    vmin, vmax = 1, max([torch.max(target).item(), torch.max(preds).item()])
    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    extent = [6578.137, 8378.136999999999, 9.999999999999999e-06, 10.0]
    
    # Create images for the first and second distributions
    im1 = ax1.imshow(target[0, :, :], aspect='auto', norm=log_norm, extent = extent)
    im2 = ax2.imshow(preds[0, :, :], aspect='auto', norm=log_norm, extent = extent)
    im3 = ax3.imshow(abs_diff[0, :, :], aspect='auto', norm=log_norm, extent = extent)    

    # Add colorbars
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    # Add titles
    ax1.set_title("MOCAT-MC")
    ax2.set_title("MOCAT-ML")
    ax3.set_title("Absolute Difference")
    ax4.set_title("Model Configuration")

    # Add x and y labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(r"$r_p$ [km]")
        ax.set_ylabel("A/m [m2/kg]")

    # Display model configuration in the third subplot
    model_config = {key:config[key] for key in ['horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'key', 'loss', 'log', 'average']}
    config_text =  "\n\n" + "\n".join([f"{key}: {value}" for key, value in model_config.items()])
    ax4.text(0.05, 0.3, config_text, fontsize=8) 
    ax4.axis('off') 

    n_iter = 2436//(config.lookback  + config.gap) - math.ceil(config.lookback/(config.lookback + config.gap))

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



if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--lookback", type = int, default = 50)
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 100) 
    parser.add_argument("--bs", type = int, default = 30)
    parser.add_argument("--n_epoch", type = int, default = 30)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--log", type = int, default = 1)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    parser.add_argument("--guidance", type = int, default = 0)  # whether or not to use guidance (parameters added along image channel)
    parser.add_argument("--save_model", type = int, default = 0) # whether or not to save the model

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)

    # Settings
    model_type = 'convgru'
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)

    config_base['convgru']['n_in'] = 3 if args.guidance else 1
    config_base['convgru']['n_out'] = 1

    config = AttrDict(config_base)

    keys = ['horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'log', 'average', 'save_model', 'guidance']  
    for key in keys:
        config[key] = arg_dict[key]

    # Manually set grid size here
    config['ip'] = [1, 10]
    config['lr'] = [1, 10]

    print("CONFIG \n", json.dumps(config, indent=4))

    train(config)  
