import sys, json, os
sys.path.append('..')
from tsai.utils import yaml2dict
import argparse, torch, h5py, torchvision, datetime

from fastcore.all import *
from fastai.vision.all import *

from tqdm import tqdm 
import matplotlib.colors as mcolors
from nbs.utils import *
from train_script_v1  import get_data
from loss_functions import *


DATA_PATH = "/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml"
N, KEY, LKB = 2436, 'comb_Am_rp', 50
NUMBER_SIZE, FONT_SIZE, TITLE_SIZE = 20, 20, 26


def get_pred(cfg, ds):
    guide, horizon = cfg.guidance, cfg.horizon
    model = StackUnstack(SimpleModel(**cfg.convgru)).to(default_device())
    model_path = f"ip_[1, 10]_lr_[1, 10]_guidance_{guide}_horizon_{horizon}_n_epochs_30.pth" # add model path 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        # Iterative forecasting
        val = get_data(ds).mean(axis=0, keepdims=True) 
        val = np.expand_dims(val, axis=2) 
        val = np.log(val+1)    
        if guide:
            params = [int(i) for i in ds.split('x')[1:]]
            val = add_guidance(val, params)
        val = torch.tensor(val[0]).float() 

        initial_seq = val[:horizon].to(default_device())
        inp = tuple(i.unsqueeze(0) for i in initial_seq)
        preds = [initial_seq[:,:1,:,:].cpu().detach()]
        
        n_iter = N//horizon
        for iter in range(1, n_iter+1):
            p = model(inp)
            preds.append(torch.vstack(p).cpu().detach())

            if iter==n_iter:
                break

            if guide:
                target = val[iter*horizon:(iter+1)*horizon].to(default_device())
                guidance = target[:,1:,:,:] 
                inp = tuple(torch.cat([p[i], guidance[i:i+1]], axis=1)  for i in range(horizon))

            else:
                inp = p
            
        preds = torch.vstack(preds)[:N]
        preds = torch.exp(preds)-1

        if guide:
            val = val[:,:1,:,:]

        val = torch.exp(val)-1
    return val, preds



def get_snapshots(ds, cfg, log):  
    guide = cfg.guidance

    horizons, years = [50], [5, 10, 20, 40, 70, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    snapshots = []
    for horizon in horizons:
        cfg.horizon = horizon
        val, preds = get_pred(cfg, ds)
        snapshots.append(preds[indices].squeeze())

    # Plotting and saving 
    val = get_data(ds).mean(axis=0) 
    if log==1:
        val = np.log(val+1)

    fig, axes = plt.subplots(len(horizons)+1, len(years), figsize=(16, 10))
    fig.suptitle("Predictions by Horizon and Year", fontsize=25)

    for i, preds_at_horizon in enumerate(snapshots):  # Each horizon's predictions
        for j, pred in enumerate(preds_at_horizon):  # Each year
            ax = axes[i, j]
            if log==1:
                pred = torch.log(pred +  1)
            ax.imshow(pred.numpy(), aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i == 0:
                ax.set_title(f'Year {years[j]}', fontsize=FONT_SIZE)
            if j == 0:
                ax.set_ylabel(f'Horizon {horizons[i]}', fontsize=FONT_SIZE-3)


    for j, _ in enumerate(years):
        ax = axes[-1, j]
        ax.imshow(val[indices[j]], aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel('Original Data', fontsize=FONT_SIZE-3)


    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.savefig(f"td1/snapshots{ds}_guide_{guide}_{log}.png", dpi=300)
    plt.show()



def plot_smapes(config):

    years = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    with torch.no_grad():
        ip_min, ip_max = config.ip
        lr_min, lr_max = config.lr

        datasets = [f'x{i}x{j}' for i in range(ip_min, ip_max+1) for j in range(lr_min, lr_max+1)]

        old, new = oldSMAPE(log=0), newSMAPE(log=0)
        new_smapes, old_smapes = [], []

        for ds in tqdm(datasets):
            val, preds = get_pred(config, ds)
            val, preds = val[indices], preds[indices]

            new_smapes.append([new(p, v).item() for p,v in zip(preds, val)])
            old_smapes.append([old(p, v).item() for p,v in zip(preds, val)])

        # Plotting 
        date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
        os.makedirs(f"td1/{date}")
        
        ip_vals = list(range(ip_min, ip_max+1))
        lr_vals = list(range(lr_min, lr_max+1))

        for year_idx in range(len(years)):
            names = ['new', 'old']
            for smapes, name in zip([new_smapes, old_smapes], names):
                metrics = np.array(smapes).reshape(len(ip_vals), len(lr_vals), len(years))[:, :, year_idx]

                # Create the heatmap
                plt.figure(figsize=(12, 12))
                heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto', vmin=0, vmax=2, origin='lower')

                # Add colorbar
                cbar = plt.colorbar(heatmap)
                cbar.ax.tick_params(labelsize=20) 

                # Add labels, title, and ticks
                plt.title(f'Year {years[year_idx]}', fontsize=30)
                plt.xlabel('Init population', fontsize=27)
                plt.ylabel('Launch rate', fontsize=27)
                plt.xticks(np.arange(len(ip_vals)), ip_vals, fontsize=27)
                plt.yticks(np.arange(len(lr_vals)), lr_vals, fontsize=27)
                plt.tight_layout()
                plt.savefig(f"td1/{date}/{name}_smape_{years[year_idx]}_{config.guidance}.png")
                plt.show()
                plt.close()



def plot_losses_long_term(ds, cfg):

    all_losses = []
    for horizon in [10, 25, 50]:
        cfg.horizon = horizon
        val, preds = get_pred(cfg, ds)
        losses = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]
        all_losses.append(losses)
    
    fig, ax = plt.subplots(figsize=(8, 6))  # One plot only

    for horizon, losses in zip([10, 25, 50], all_losses):
        losses = losses[::50]
        ax.plot(np.linspace(0, 100, len(losses)), losses, label=f'Horizon {horizon}')

    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("MAE loss", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"td1/{ds}_horizons_losses.png", dpi=300)
    plt.show()



def plot_loss_vs_MC(ds, cfg):
    cfg.guidance = 0
    config_base['convgru']['n_in'] = 1
    config_base['convgru']['n_out'] = 1

    val, preds = get_pred(cfg, ds)
    losses_no_guidance = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]

    cfg.guidance = 1
    config_base['convgru']['n_in'] = 3 
    config_base['convgru']['n_out'] = 1

    val, preds = get_pred(cfg, ds)
    losses_with_guidance = [nn.L1Loss()(p,v).item() for p, v in zip(preds, val)]

    fig, ax = plt.subplots(figsize=(8, 6))  
    ax.plot(np.linspace(0, 100, N), losses_no_guidance, label=f'Without guidance')
    ax.plot(np.linspace(0, 100, N), losses_with_guidance, label=f'With guidance')
    ax.set_title(f"Forecasting error over time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("MAE loss", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{ds}_losss_comparison.png", dpi=300)
    plt.show()




def plot_number_of_objects(ds, cfg):

    val, preds = get_pred(cfg, ds)
    val, preds = val.squeeze(), preds.squeeze()
    val_objects, preds_objects = val.sum(axis=(-1,-2)), preds.sum(axis=(-1,-2))

    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())

    _, ax = plt.subplots(figsize=(8, 6))  

    ax.plot(np.linspace(0, 100, N), val_objects, label='Target')
    ax.plot(np.linspace(0, 100, N), preds_objects, label='Prediction')
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"td1/{ds}_total_number_objects_{cfg.guidance}.png", dpi=300)
    plt.show()




if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--guidance", type = int, default = 0)  # whether or not to use guidance (parameters added along image channel)


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

    keys = ['horizon',  'guidance']  
    for key in keys:
        config[key] = arg_dict[key]

    # Manually set grid size here
    config['ip'] = [1, 10]
    config['lr'] = [1, 10]
    config['log'] = 1

    print("CONFIG \n", json.dumps(config, indent=4))

    ds = 'x9x10'
    plot_number_of_objects(ds, config)
