import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.utils import *
from mocatml.models.conv_rnn import *
from mygrad import sliding_window_view
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse, os, h5py

my_setup()

def get_loss_func_and_metrics(config):
    if config.partial_loss is not None:
        if config['loss'] == 'mse':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
        if config['loss'] == 'mae':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=L1LossFlat())
        if config['loss'] == 'huber':
            loss_func = PartialStackLoss(config.partial_loss, loss_func=nn.HuberLoss())

        full_loss = StackLoss()
        full_loss.__name__ = "full_loss"
        metrics = [full_loss] 

    else:
        if config['loss'] == 'mse':
            loss_func = StackLoss(MSELossFlat())
        if config['loss'] == 'mae':
            loss_func = StackLoss(L1LossFlat())
        if config['loss'] == 'huber':
            loss_func = StackLoss(nn.HuberLoss())
        metrics = []

    return loss_func, metrics


def train(config):
    # only implemented for convgru (add more architectures)    
    
    loss_func, metrics = get_loss_func_and_metrics(config)

    # model setup
    dls = get_ds_grid(config)
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
    cbs = L() + wandbc
    learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)    
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()
    
    # training 
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    # learn.fit(config.n_epoch, 1e-1)
    # learn.eval()
    # plot_preds(learn, config, X, X_sw, dataset)
    return learn

def get_ds_multi(ds_name, config):
    main_path = '/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/'
    path = f'{main_path}{ds_name}/TLE_density_all.mat'
    file = h5py.File(path, 'r')
    data = np.concatenate([np.expand_dims(np.array(file[key]), axis=2) for key in ['comb_Am_rp', 'comb_Am_ra']], axis=2)
    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,5,2,3,4)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw

def get_ds_single(ds_name, config):
    main_path = '/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/'
    path = f'{main_path}{ds_name}/TLE_density_all.mat'
    file = h5py.File(path, 'r')
    data = np.array(file['comb_Am_rp'][:])

    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw


def get_ds_grid(config):
    xPop, xLaunch = [int(i) for i in config['ds'].split('x')[1:]]
    train_data, train_data_sw = [], []
    
    for xP in range(xPop-config['width'], xPop+config['width']+1):
        for xL in range(xLaunch-config['width'], xLaunch+config['width']+1):
            if xP < 0 or xL < 0 or xP > 15 or xL > 15: continue
            if xP+xL == 0 or (xP == xPop and xL == xLaunch): continue 
            
            ds_name = f'x{xP}x{xL}'
            data, data_sw = get_ds_multi(ds_name, config) if config['multi'] else get_ds_single(ds_name, config)
            train_data.append(data)
            train_data_sw.append(data_sw)

    train_data, train_data_sw = np.vstack(train_data), np.vstack(train_data_sw)            
    val_data, val_data_sw = get_ds_multi(ds_name, config) if config['multi'] else get_ds_single(ds_name, config)
    
    print(f"train data shape {train_data.shape} val data shape {val_data.shape}")

    train_ds = DensityData(train_data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    val_ds = DensityData(val_data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    
    # normalization!!
    
    mocat_stats = np.mean(train_data), np.std(train_data)

    train_tl = TfmdLists([i for i in range(train_data.shape[1])], DensityTupleTransform(train_ds))
    valid_tl = TfmdLists([i for i in range(val_data.shape[1])], DensityTupleTransform(val_ds))
    
    
    dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config.bs, device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.normalize else None,
                        num_workers=config.num_workers)
    
    return dls


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if __name__ == "__main__":
    
    # TODO - add wandb implementation and more architectures

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds", type = str, default = "x5x5") 
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4)
    parser.add_argument("--lookback", type = int, default = 4)
    parser.add_argument("--stride", type = int, default = 8)
    parser.add_argument("--partial_loss", type = int, default = 1) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--width", type = int, default = 1)
    parser.add_argument("--multi", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--loss", type = str, default='mse')

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'width', 'multi', 'loss']:
        config[key] = arg_dict[key]
        
    if config['multi'] == 1:
        config_base[model_type]['n_in'] = 2
    
    xPop, xLaunch = [int(i) for i in args.ds.split('x')[1:]]
    if xPop < 0 or xPop > 15 or xLaunch < 0 or xLaunch > 15: 
        raise ValueError("xPop and xLaunch should be in the range of [0:15]")
    
    train(config)
