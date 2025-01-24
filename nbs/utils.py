import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
import os, h5py, torch
import numpy as np

import torch
from torchvision import transforms

from loss_functions import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation


KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]

def get_dataset(ds_name, config):  
    path = f'{config.data.path}/TLE_density_all_{ds_name}.mat' # Use absolute path 
    mat = h5py.File(path, 'r')
    data = np.array(mat[config.key])[:, :config.sel_steps]
    if config.sample:
        num_sim, timesteps = data.shape[:2]
        transform = transforms.Compose([    
            transforms.Resize((32, 32))
        ])  
        
        data_reshaped = torch.tensor(data).view((-1, 1, 36, 99))
        transformed_data = [transform(sample) for sample in data_reshaped]
        data = torch.stack(transformed_data, dim=0)
        data = np.array(data.reshape(num_sim, timesteps, 32, 32))

    if config.average:
        averages = []
        for i in range(5):
            x = np.expand_dims(np.mean(data[10*i:10*(i+1)], axis=0), 0)
            averages.append(x)
        data = np.vstack(averages)

    if config.log:
        data = np.log(data + 1)

    # data = data[:20]  # only use 20 simulations from the dataset
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw



def get_dataloader(dataset, config):
    data, data_sw = get_dataset(dataset, config)
    splits = RandomSplitter()(data) #valid_pct=0.2,
    ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    samples_per_simulation = data_sw.shape[0]//(data.shape[0])
    train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
    valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

    print(len(train_idxs), len(valid_idxs))

    mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))

    train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
    valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
    
    dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config['bs'], device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.normalize else None,
                        num_workers=config.num_workers)
    
    return dls, splits, data 



def get_dataloader_from_dslist(ds_list, config):
    train_dls, valid_dls, splits, Xs = [], [], [], []
    for ds_name in ds_list:
        X, X_sw = get_dataset(ds_name, config)
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
                        config.normalize else None,
                        num_workers=config.num_workers)

    return dls, splits, Xs


class MOCAT_Dataset(Dataset):
    def __init__(self, data_sw, lkb=4, hrzn=4, gap=0):
        super(MOCAT_Dataset, self).__init__()

        self.data = data_sw
        self.lkb = lkb
        self.hrzn = hrzn
        self.gap = gap
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        data2, data3 = x[:self.lkb], x[self.lkb+self.gap:self.lkb+self.gap+self.hrzn]
        return data2, data3
        


def calculate_sample_idxs(simulation_idxs, samples_per_sim):
    indices = []
    for sim in simulation_idxs:
        start_idx = sim * samples_per_sim
        end_idx = start_idx + samples_per_sim
        indices.extend(range(start_idx, end_idx))
    return indices



def get_dataloader_diffusion(config):

    ds = config.ds
    path = f'/mnt/data/sumiya/mocat-ml/data/'
    filename = f"TLE_density_all_{ds}.mat"
    file = h5py.File(path + filename, 'r')  
    data = np.array(file[config.key])
    lkb, hrzn, gap, stride, sample = config.lookback, config.horizon, config.gap, config.stride, config.sample

    if sample:
        if filename[:-4] + "_sampled.npy" in os.listdir(path+f"sampled_{config.key}/"):
            data = np.load(path+f"sampled_{config.key}/"+filename[:-4] + "_sampled.npy")

        else:
            # slow
            num_sim, timesteps = data.shape[:2]
            data_tensor = torch.tensor(data).view(-1, 1, 36, 99)
            transform = transforms.Compose([transforms.Resize((32, 32))])
            transformed_data = transform(data_tensor)
            data = transformed_data.view(num_sim, timesteps, 32, 32).numpy()
            np.save(path + f"sampled_{config.key}/"+filename[:-4] + "_sampled.npy", data)

    if config.average:
        averages = []
        for i in range(5):
            x = np.expand_dims(np.mean(data[10*i:10*(i+1)], axis=0), 0)
            averages.append(x)
        data = np.vstack(averages)

    if config.log:
        data = np.log(data + 1)

    if config.normalize:
        num_sim, num_timesteps, _, _ = data.shape
        normalized_data = np.zeros_like(data)
        max_value = np.max(data)
        min_value = np.min(data)
        for s in range(num_sim):
            for t in range(num_timesteps):
                timestep_data = data[s][t]
                normalized_data[s][t] = (timestep_data - min_value) / (max_value - min_value) #0 to 1
                normalized_data[s][t] = normalized_data[s][t] * 2 - 1 # -1 to 1

        data = normalized_data
    
    data_sw = np.lib.stride_tricks.sliding_window_view(data, lkb + hrzn + gap, axis=1)[:,::stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])

    print(data_sw.shape, data.shape)

    splits = RandomSplitter()(data) #valid_pct=0.2,
    samples_per_simulation = data_sw.shape[0]//(data.shape[0])
    train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
    valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

    print(len(train_idxs), len(valid_idxs))

    train_dataset = MOCAT_Dataset(data_sw[train_idxs], lkb=config.lookback, hrzn=config.horizon)
    val_dataset = MOCAT_Dataset(data_sw[valid_idxs], lkb=config.lookback, hrzn=config.horizon)

    bs = config.bs
    train_dl = DataLoader(train_dataset, batch_size=bs,
                            shuffle=False,
                            pin_memory=True, # pin_memory set to True
                            num_workers=12,
                            prefetch_factor=4,
                            drop_last=False)

    val_dl = DataLoader(val_dataset, batch_size=bs,
                            shuffle=False,
                            pin_memory=True, # pin_memory set to True
                            num_workers=12,
                            prefetch_factor=4,  
                            drop_last=False)
    
    print('Train loader and Valid loader are up!')
    return train_dl, val_dl, data, data_sw, splits, val_dataset, max_value, min_value



def get_dataset_4d(config):
    num_sim = 10
    data = np.zeros((num_sim, 2436, 16, 16, 16, 16))
    ds = config.ds
    for i in range(num_sim):
        path = f'/mnt/data/sumiya/mocat-ml/4d_16/{ds}/TLE_density_{i}.mat'
        mat = h5py.File(path, 'r')
        data[i]= np.array(mat["comb_ra_rp_inc_Am"])[:, :config.sel_steps]  
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,6,2,3,4,5)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw



def get_dataloader_4d(config):
    data, data_sw = get_dataset_4d(config)
    print(data.shape, data_sw.shape)
    
    splits = RandomSplitter()(data) #valid_pct=0.2,
    ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    samples_per_simulation = data_sw.shape[0]//(data.shape[0])
    train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
    valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

    print(len(train_idxs), len(valid_idxs))

    mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))

    train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
    valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
    
    dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config['bs'], device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.normalize else None,
                        num_workers=config.num_workers, distributed=config.num_workers>0)
    
    return dls, splits, data, data_sw


def plot_loss(recorder, skip_start=0, with_valid=True, log=False, show_epochs=False, ax=None):
    if not ax:
        ax=plt.gca()
    if log:
        ax.loglog(list(range(skip_start, len(recorder.losses))), recorder.losses[skip_start:], label='train')
    else:
        ax.plot(list(range(skip_start, len(recorder.losses))), recorder.losses[skip_start:], label='train')
    if show_epochs:
        for x in recorder.iters:
            ax.axvline(x, color='grey', ls=':')
    ax.set_ylabel('loss')
    ax.set_xlabel('steps')
    if with_valid:
        idx = (np.array(recorder.iters)<skip_start).sum()
        valid_col = recorder.metric_names.index('valid_loss') - 1 
        ax.plot(recorder.iters[idx:], L(recorder.values[idx:]).itemgot(valid_col), label='valid')
        ax.legend()
    return ax


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



def plot_preds(learn, config, X, X_sw, save_folder, years_to_plot = [1/6, 1, 2, 3, 4, 5, 10, 100]):
    train_stats = (learn.dls.train.after_batch.mean, learn.dls.train.after_batch.std)
    ds_full = DensityData(X, lbk=config.lookback, h=config.horizon)
    tl_full = TfmdLists(range(len(ds_full)), DensityTupleTransform(ds_full))
    dl_full = TfmdDL(tl_full, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
                after_batch=Normalize.from_stats(*train_stats))

    for year in years_to_plot:
        n_iter = (X.shape[1]*year)//(config.horizon*100) - 1 if year>= 1 else 1
        print(year, n_iter)

        inps, preds, targs, losses = learn.get_preds_iterative(dl=dl_full, n_iter=n_iter, track_losses=True, with_input=True)

        if not os.path.exists(save_folder+f"{year}/"):
            os.makedirs(save_folder+f"{year}")

        if year == 100:
            plt.clf()
            plt.plot(np.linspace(0, 100, losses.shape[0]), losses)
            plt.xlabel("Years")
            plt.ylabel(f"Loss ({config['loss']})")
            plt.savefig(f"{save_folder}/loss-100-years.jpg")

            with open(f"{save_folder}/loss-100-years.txt", "w") as output:
                output.write(str(losses))

        title_input = "input"
        title_pred = f"{year} year-ahead predictions Loss: {losses[-1]}" if year > 1 else f"2 month-ahead predicitons Loss: {losses[-1]}"
        title_target = f"{year} year-ahead targets" if year > 1 else f"2 month-ahead targets"

        learn.show_preds_at(0, p=preds, t=targs, inp=inps, save=True, save_path = save_folder+f"{year}/", with_targets=True, 
                        with_input=True, start_epoch=(n_iter-1)*config.horizon,
                    titles=[title_input, title_pred, title_target])
        

def plot_metrics(metric_scores_by_year, metric_scores_full, save_folder, stride=10):

    # Example variables and evaluation metrics
    launch_rates = sorted(list(set([int(l.split('x')[1]) for l in metric_scores_by_year])))[::-1]  # Columns
    init_pops = sorted(list(set([int(l.split('x')[2]) for l in metric_scores_by_year])))        # Rows

    for ii in range(10):
        metrics = np.array([
            [metric_scores_by_year[f'x{ip}x{lr}'][ii] for ip in init_pops] for lr in launch_rates
        ])

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        heatmap = plt.imshow(metrics, cmap='Blues', aspect='auto')

        # Add colorbar
        plt.colorbar(heatmap)

        # Add labels, title, and ticks
        plt.title('SMAPE', fontsize=14)
        plt.xlabel('Init population', fontsize=12)
        plt.ylabel('Launch rate', fontsize=12)
        plt.xticks(np.arange(len(init_pops)), init_pops, fontsize=10)
        plt.yticks(np.arange(len(launch_rates)), launch_rates, fontsize=10)

        # Annotate cells with metric values
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                plt.text(j, i, f'{metrics[i, j]:.3f}', ha='center', va='center', color='black')

        # Show the plot
        plt.tight_layout()
        plt.savefig(save_folder+f'/SMAPE_year_{(ii+1)*10}.jpg')
        plt.show()



    # Create the heatmap gif
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create images for the first and second distributions
    metrics = np.array([
        [metric_scores_full[f'x{ip}x{lr}'][0] for ip in init_pops] for lr in launch_rates
    ])

    n = len(metric_scores_full[f'x{init_pops[0]}x{launch_rates[0]}'])

    # Create the heatmap
    im1 = ax.imshow(metrics, cmap='Blues', aspect='auto')
    fig.colorbar(im1, ax=ax)
    ax.set_xlabel('Init population', fontsize=10)
    ax.set_ylabel('Launch rate', fontsize=10)
    ax.set_xticks(np.arange(len(init_pops)), init_pops, fontsize=10)
    ax.set_yticks(np.arange(len(launch_rates)), launch_rates, fontsize=10)

    # Initialize annotations
    annotations = []
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            annotations.append(ax.text(j, i, f'{metrics[i, j]:.3f}', ha='center', va='center', color='black'))

    # Define the update function
    def update(frame):
        """Update the heatmap and annotations for each animation frame."""
        # Extract the data for the current frame
        metrics = np.array([
            [metric_scores_full[f'x{ip}x{lr}'][frame * stride] for ip in init_pops] for lr in launch_rates
        ])

        im1.set_array(metrics)

        # Update text annotations
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                annotations[i * metrics.shape[1] + j].set_text(f'{metrics[i, j]:.3f}')
        
        fig.suptitle(f'SMAPE year: {round(frame * stride * 100 / n)}')
        return [im1] + annotations + [fig]

    # Create the figure and axis
    ani = FuncAnimation(fig, update, frames=n//stride, interval=50, blit=True) 

    # Save the animation as a GIF
    ani.save(f'{save_folder}SMAPE_evolution.gif', writer='imagemagick', fps=20) 
    plt.show()


def save_results(config, lookback, horizon, preds, save_to, epoch, long_term=False, iter_num=None):
    n, figsize = config.lookback, (4, 3)
    fig, axs = plt.subplots(nrows = 4, ncols=n, figsize=(figsize[0]*n, 5*figsize[1]), squeeze=False)  

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    for i, lkb, hor, pred in zip(range(config.lookback), lookback, horizon, preds):
        if config.log:
            lkb, hor, pred = torch.exp(lkb)-config.eps, torch.exp(hor)-config.eps, torch.exp(pred)-config.eps

        axs[0, i].imshow(lkb.detach().cpu(), aspect = 'auto')
        axs[0, i].set_title('Input')

        axs[1, i].imshow(hor.cpu(), aspect = 'auto')
        axs[1, i].set_title('Target')
            
        axs[2, i].imshow(pred.detach().cpu().numpy(), aspect = 'auto')
        axs[2, i].set_title('Prediction')

        diff_pred_lkb = torch.abs(pred - hor)
        abs_diff = round(torch.sum(torch.abs(diff_pred_lkb)).item()/torch.sum(torch.abs(hor)).item(), 2)
        pcm  = axs[3, i].imshow(diff_pred_lkb.detach().cpu().numpy(), aspect = 'auto', cmap = 'coolwarm')
        axs[3, i].set_title('Pred/Target Percent Error: ' + str(abs_diff))
                            
        fig.colorbar(pcm, ax=axs[3, i])

        for ax in axs:
            ax[i].axis('on')

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95) 
    # plt.title(f"One step ahead prediction after {epoch} epochs")
    plt.tight_layout()
    if long_term:
        plt.savefig(f"{save_to}/{iter_num}")
        
    else:
        plt.savefig(f"{save_to}/1 step prediction after {epoch} epochs")
    plt.show()
    plt.close()

