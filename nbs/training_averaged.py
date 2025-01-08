import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from mygrad import sliding_window_view
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, torch, time
import numpy as np

from loss_functions import *
from utils import *

my_setup()

def train_on_dataset(config):
    # only implemented for convgru (add more architectures)
    dls, splits, X, X_sw = get_dataloader_averaged(config)
    loss_func, metrics = get_loss_func_and_metrics(config)

    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
    cbs = L() + wandbc
    learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
    learn.splits = splits # This is needed for the evaluation notebook
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
    
    # training 
    print("MODEL SIZE: ", get_n_params(learn), "\n")

    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    plot_preds(learn, config, X, X_sw, save_folder=config.save_folder)

    return learn



def get_dataloader_averaged(config):
    data, data_sw = get_dataset_averaged(config)
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



def get_dataset_averaged(config):
    path = "/mnt/data/sumiya/mocat-ml/data/"
    ds_name = f"TLE_density_all_{config.ds}.mat"
    ds = path+ds_name
    if ds_name not in os.listdir(path):
        raise "DATASET not found"
    
    mat = h5py.File(ds, 'r')
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
        print(data.shape)

    averages = []
    for i in range(5):
        x = np.expand_dims(np.mean(data[10*i:10*(i+1)], axis=0), 0)
        averages.append(x)
    
    data = np.vstack(averages)
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw



def plot_preds(learn, config, X, X_sw, save_folder):

    train_stats = (learn.dls.train.after_batch.mean, learn.dls.train.after_batch.std)
    ds_full = DensityData(X, lbk=config.lookback, h=config.horizon)
    tl_full = TfmdLists(range(len(ds_full)), DensityTupleTransform(ds_full))
    dl_full = TfmdDL(tl_full, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
                after_batch=Normalize.from_stats(*train_stats))

    for year in config.recon_years:
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



if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds", type = str, default = "x8x8", help = "dataset to train on")     
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4) 
    parser.add_argument("--lookback", type = int, default = 4) 
    parser.add_argument("--stride", type = int, default = 8) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--sample", type = int, default = 0) # whether or not to sample to 32x32
    parser.add_argument("--recon_years", type = list, default=[10, 20, 30, 40, 50, 60, 70, 80 ,90, 100])


    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'sample', 'recon_years']:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    name = f"{args.ds}_averaged/n_epoch_{config.n_epoch}_sample_{config.sample}_hor_lkb_{config.horizon}_str_{config.stride}_bs_{config.bs}"
    if config.partial_loss is None:
        if config['loss'] != 'mbd':
            save_folder = f"plots/{config.loss}/{name}/" 
        else: 
            save_folder = f"plots/mbd_alpha_{config.alpha}/{name}/"
    else:
        if config['loss'] != 'mbd':
            save_folder = f"plots/{config.loss}/{name}_partial_1/"
        else:
            save_folder = f"plots/mbd_alpha_{config.alpha}/{name}_partial_1/"

    config['save_folder'] = save_folder

    train_on_dataset(config)

