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


def train(config):
    # only implemented for convgru (add more architectures)    
    if config.partial_loss is not None:
        loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
        full_loss = StackLoss()
        full_loss.__name__ = "full_loss"
        metrics = [full_loss] # [StackLoss()]
    else:
        loss_func = StackLoss(MSELossFlat())
        metrics = []

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


def get_dataset(ds_name, config):
    main_path = '/home/gridsan/ssarangerel/orbitalrisk_MC/supercloud_runs/combined_ds_mocatml/'
    path = f'{main_path}{ds_name}/TLE_density_all.mat'
    data = h5py.File(path, 'r')
    data = np.array(data['comb_Am_rp'][:])

    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw


def get_ds_grid(config):
    
    train_data, train_data_sw, val_data, val_data_sw = None, None, None, None
    
    for xPop in range(config['train_xPop_1'], config['train_xPop_2']+1):
        for xLaunch in range(config['train_xLaunch_1'], config['train_xLaunch_2']+1):
            if skip(xPop, xLaunch, config): continue
            
            ds_name = f'x{xPop}x{xLaunch}'
            if train_data is None: 
                train_data, train_data_sw = get_dataset(ds_name, config)

            else:
                ds, ds_sw = get_dataset(ds_name, config)
                train_data = np.vstack([train_data, ds])
                train_data_sw = np.vstack([train_data_sw, ds_sw])
                
            print(f"train: x{xPop}x{xLaunch}", train_data.shape, train_data_sw.shape)
            
            
    for xPop in range(config['val_xPop_1'], config['val_xPop_2']+1):
        for xLaunch in range(config['val_xLaunch_1'], config['val_xLaunch_2']+1):
            if xPop + xLaunch == 0: continue
            
            ds_name = f'x{xPop}x{xLaunch}'
            if val_data is None: 
                val_data, val_data_sw = get_dataset(ds_name, config)

            else:
                ds, ds_sw = get_dataset(ds_name, config)
                val_data = np.vstack([val_data, ds])
                val_data_sw = np.vstack([val_data_sw, ds_sw])
                
            print(f"val: x{xPop}x{xLaunch}", val_data.shape, val_data_sw.shape)
    
    
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


def skip(xPop, xLaunch, config):
    if xPop + xLaunch == 0: return 1
            
    if xPop >= config['val_xPop_1'] and xPop <= config['val_xPop_2']:
        if xLaunch >= config['val_xLaunch_1'] and xLaunch <= config['val_xLaunch_2']:
            return 1
    return 0

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
    parser.add_argument("--train_ds_grid", type = str, default = "1,3,1,3") 
    parser.add_argument("--val_ds_grid", type = str, default = "2,2,2,2")
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 4)
    parser.add_argument("--lookback", type = int, default = 4)
    parser.add_argument("--stride", type = int, default = 8)
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 20)
    parser.add_argument("--sel_steps", type = int, default = None)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps']:
        config[key] = arg_dict[key]
    
    train_ranges, val_ranges = args.train_ds_grid.split(','), args.val_ds_grid.split(',')
    if len(train_ranges) != 4 or len(val_ranges) != 4:
        raise ValueError("Wrong format")
    
    for i, j in zip(train_ranges, ['train_xPop_1', 'train_xPop_2', 'train_xLaunch_1', 'train_xLaunch_2']):
        config[j] = int(i)
        
    for i, j in zip(val_ranges, ['val_xPop_1', 'val_xPop_2', 'val_xLaunch_1', 'val_xLaunch_2']):
        config[j] = int(i)
        
    train(config)
    
    # print("CONFIG \n", json.dumps(config, indent=4))  
