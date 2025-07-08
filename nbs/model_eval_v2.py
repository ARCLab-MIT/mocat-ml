import sys, json, os
sys.path.append('..')
from tsai.utils import yaml2dict
import argparse, torch, h5py, torchvision, datetime

from fastcore.all import *
from fastai.vision.all import *

from tqdm import tqdm 
import matplotlib.colors as mcolors
from nbs.utils import *
from loss_functions import *
from inference import get_data, get_pred, get_pred_channel, get_snapshots, get_snapshots_combined


DATA_PATH = "/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam" #TODO change it accordingly
N, KEY, LKB = 2436, 'comb_Am_rp', 50
NUMBER_SIZE, FONT_SIZE, TITLE_SIZE = 20, 20, 26



# Plotting SMAPEs over time
def smapes_over_time(cfg, ds_name):
    val, preds = get_pred(cfg, ds_name)
    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)

    new_smapes = [new(p, v).item() for p,v in zip(preds, val)]
    old_smapes = [old(p, v).item() for p,v in zip(preds, val)]
    total_smapes = [total(p, v).item() for p,v in zip(preds, val)]
    
    names = ['new', 'old', 'total']
    smape_lists = [new_smapes, old_smapes, total_smapes]

    for smapes, name in zip(smape_lists, names):
        _, ax = plt.subplots(figsize=(8, 6))  
        ax.plot(np.linspace(0, 100, len(smapes)), smapes, linewidth=2)
        ax.set_title(f"SMAPE Over Time", fontsize=FONT_SIZE)
        ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
        ax.set_ylabel("SMAPE", fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
        plt.tight_layout()
        plt.savefig(f"evals/{name}_smape_{ds_name}.png") #change this accordingly
        plt.close()




def plot_smapes(cfg, ip, lr):
    """
    Plot SMAPEs across pmd and cam rates on a fixed initial population and a fixed launch rate
    """
    years = [10, 40, 50, 70, 100]
    indices = [int(y/100 * N) - 1 for y in years]

    d = f'x{ip}x{lr}'
    pmds = [0.95, 0.96, 0.97, 0.98, 0.99]
    cams = [0.95, 0.96, 0.97, 0.98, 0.99]
    datasets = [d + f'x{pmd:.2f}x{cam:.2f}' for pmd in pmds for cam in cams]

    old, total, new = oldSMAPE(log=0), totalSMAPE(log=0), newSMAPE(log=0)
    new_smapes, old_smapes, total_smapes = [], [], []

    for ds in tqdm(datasets):
        val, preds = get_pred(cfg, ds)
        val, preds = val[indices], preds[indices]

        new_smapes.append([new(p, v).item() for p,v in zip(preds, val)])
        old_smapes.append([old(p, v).item() for p,v in zip(preds, val)])
        total_smapes.append([total(p, v).item() for p,v in zip(preds, val)])


    new_smapes = np.array(new_smapes).reshape(len(cams), len(pmds), -1)
    old_smapes = np.array(old_smapes).reshape(len(cams), len(pmds), -1)
    total_smapes = np.array(total_smapes).reshape(len(cams), len(pmds), -1)
    

    for year_idx in range(len(years)):
        names = ['new', 'old', 'total']
        for smapes, name in zip([new_smapes, old_smapes, total_smapes], names):
            metrics = smapes[:, :, year_idx]

            # Create the heatmap
            plt.figure(figsize=(12, 12))
            heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto', vmin=0, vmax=2, origin='lower')

            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.ax.tick_params(labelsize=20) 

            # Add labels, title, and ticks
            plt.title(f'Year {years[year_idx]}', fontsize=30)
            plt.xlabel('CAM', fontsize=27)
            plt.ylabel('PMD', fontsize=27)
            plt.xticks(np.arange(len(pmds)), pmds, fontsize=27)
            plt.yticks(np.arange(len(cams)), cams, fontsize=27)
            plt.tight_layout()

            # Change accordingly
            if cfg.guidance == 1:
                plt.savefig(f"td2/smapes/guidance/{name}_smape_{years[year_idx]}_x{ip}x{lr}.png")
            else:
                plt.savefig(f"td2/smapes/no_guidance/{name}_smape_{years[year_idx]}_x{ip}x{lr}.png")

            plt.show()
            plt.close()




def plot_number_of_objects(ds, cfg):

    val, preds = get_pred(cfg, ds)
    val, preds = val.squeeze(), preds.squeeze()
    val_objects, preds_objects = val.sum(axis=(-1,-2)), preds.sum(axis=(-1,-2))

    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f'zevals/{date}')

    _, ax = plt.subplots(figsize=(8, 6))  #
    ax.plot(np.linspace(0, 100, N), val_objects, label='Target')
    ax.plot(np.linspace(0, 100, N), preds_objects, label='Prediction')

    ax.set_title(f"Total Number of Objects Over Time", fontsize=FONT_SIZE)
    ax.set_xlabel("Time (years)", fontsize=FONT_SIZE)
    ax.set_ylabel("Number of objects", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=NUMBER_SIZE)
    ax.legend(fontsize=NUMBER_SIZE)
    plt.tight_layout()
    plt.savefig(f"zevals/{date}/{ds}_total_number_objects_{cfg.guidance}.png", dpi=300)
    plt.show()




if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Model evaluation")

    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--guidance", type = int, default = 0)  
    parser.add_argument("--dual", type = int, default = 0)  

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)

    # Settings
    model_type = 'convgru'
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)

    config = AttrDict(config_base)

    keys = ['horizon',  'guidance', 'dual']  
    for key in keys:
        config[key] = arg_dict[key]

    config['log'] = 1
    
    ds = "x2x2x0.99x0.99"

    plot_number_of_objects(ds, config)

