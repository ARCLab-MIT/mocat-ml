import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from mygrad import sliding_window_view
import os, h5py, torch
import numpy as np

import torch
from torchvision import transforms

from loss_functions import *
import torch.nn.functional as F


KEYS=["comb_Am_inc", "comb_Am_ra", "comb_Am_rp", "comb_inc_ra", "comb_inc_rp", "comb_ra_rp"]

def get_dataset(ds_name, config):  
    path = f'{config.data.path}/TLE_density_all_{ds_name}.mat'

    if config.downsample == 1:
        combined_data = np.zeros((50, 2436, len(KEYS), 36, 36))

        #TODO make this faster
        file = h5py.File(path, 'r')
        for nch, key in enumerate(KEYS):
            data = np.array(file[key])[:, :config.sel_steps]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    combined_data[i, j, nch] = data[i, j] if data[i, j].shape == (36, 36) else downsample_avg(data[i, j])

        data_sw = np.lib.stride_tricks.sliding_window_view(combined_data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
        data_sw = data_sw.transpose(0,1,5,2,3,4)
        data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
        print(data_sw.shape)
        return combined_data, data_sw

    else:
        mat = h5py.File(path, 'r')
        data = np.array(mat[config.key])[:, :config.sel_steps]
        d = np.array(mat[config.key])[:, :config.sel_steps]
        if config.sample:
            num_sim, timesteps = data.shape[:2]
            transform = transforms.Compose([    
                transforms.Resize((32, 32))
            ])  
            
            data_reshaped = torch.tensor(data).view((-1, 1, 36, 99))

            transformed_data = []
            for sample in data_reshaped:
                transformed_sample = transform(sample)
                transformed_data.append(transformed_sample)

            data = torch.stack(transformed_data, dim=0)
            data = np.array(data.reshape(num_sim, timesteps, 32, 32))
            print(data.shape)

        data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
        data_sw = data_sw.transpose(0,1,4,2,3)
        data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
     
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        # extent=[np.min(mat['rp_disc']), np.max(mat['rp_disc']), np.min(mat['Am_disc']), np.max(mat['Am_disc'])]
        # axes[0].imshow(d[0][2000], extent = extent, aspect='auto')
        # axes[0].set_xlabel('rp [km]')  
        # axes[0].set_ylabel('Am [m2/kg]') 
        # axes[0].set_title('Original')
        
        # axes[1].imshow(np.array(transformed_data[2000][0]), extent = extent, aspect='auto')
        # axes[1].set_xlabel('rp [km]') 
        # axes[1].set_ylabel('Am [m2/kg]') 
        # axes[1].set_title('Sampled to 32x32')
        # fig.tight_layout()

        # plt.savefig('transformed.png')

        return data, data_sw


def get_dataloader(dataset, config):
    data, data_sw = get_dataset(dataset, config)
    print(data_sw.shape, data.shape)
    
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



def downsample_and_fill(data, target_shape=(36, 36)):
  if data.shape != target_shape:
    return downsample_avg(data)
  else:
    return data



def downsample_avg(image):
  return downsample_avg_numpy_divisible(pad_with_zeros(image))


def downsample_avg_numpy_divisible(image):
    in_height, in_width = image.shape
    h_factor, w_factor = in_height // 36, in_width // 36
    blocks = image.reshape(in_height // h_factor, h_factor, in_width // w_factor, w_factor)
    downsampled_image = np.mean(blocks, axis=3)

    if downsampled_image.shape[0] != 36:
        return np.mean(downsampled_image, axis=0)
    return downsampled_image



def pad_with_zeros(image):
  image_height, image_width = image.shape
  if image_width==180: return image

  target_height, target_width = 36, 108

  # Calculate the amount of padding required on each side
  pad_left = (target_width - image_width) // 2
  pad_right = target_width - image_width - pad_left
  pad_top = pad_bottom = 0  # Assuming no padding on top and bottom (can be adjusted)

  padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
  padded_image = np.pad(image, padding_widths, mode='constant', constant_values=0)
  return padded_image

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

