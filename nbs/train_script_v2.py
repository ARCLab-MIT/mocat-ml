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
import wandb, json, argparse, os, torch, random, datetime, h5py
import numpy as np

from loss_functions import *
from util_train import add_guidance, get_n_params, save_training_metrics


RANDOM_DATA_PATH = '/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/new_dataset/'
VALIDATIOON_DATA_PATH = '/home/gridsan/ssarangerel/mocatmc-pmd-cam/orbitalrisk_MC/supercloud_runs/combined-pmd-cam/'
KEY, N, NORM = 'comb_Am_rp', 2436, 13


def data_to_data_sw(data, args, params):
    """
    Converts loaded data for training (num_simulations, timesteps, c, w, h => num_samples, lookback + horizon, c, w, h)
    """
    if args.dual == 0:
        if args.channel == 0:
            data = data[:,:,:,:,:1]

        elif args.channel == 1:
            data = data[:,:,:,:,1:2]

        else:
            data = data.sum(axis=-1, keepdims=True) 

    data = np.transpose(data, (0, 1, 4, 2, 3)) # (num_simulations, timesteps, c, w, h)
    _, timesteps, c, w, h = data.shape

    if args.average:
        data = data.reshape(5, -1, timesteps, c, w, h).mean(axis=1)

    if args.log:
        data = np.log(data + 1)

    if args.guidance:
        data = add_guidance(data, params)
        
    data_sw = np.lib.stride_tricks.sliding_window_view(data, args.lookback + args.horizon + args.gap, axis=1)[:,::args.stride,:]
    data_sw = data_sw.transpose(0,1,5,2,3,4)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw




def get_train_data(args, ds_name, params):
    path = f'{RANDOM_DATA_PATH}/{ds_name}/TLE_density_all.mat' 
    mat = h5py.File(path, 'r')
    data = np.array(mat[KEY]) 
    return data_to_data_sw(data, args, params)





def get_valid_data(args, params):
    ip, lr, pmd, cam = params
    ds_name = f'x{ip}x{lr}x{pmd:.2f}x{cam:.2f}'
    path = f'{VALIDATIOON_DATA_PATH}/TLE_density_all_{ds_name}.mat' 
    mat = h5py.File(path, 'r')
    data = np.array(mat[KEY])
    return data_to_data_sw(data, args, params)  




def get_dls_random(args):
    """
    Loads training data by selecting datasets randomly from RANDOM_DATA_PATH 
    """

    all_datasets = [file for file in os.listdir(RANDOM_DATA_PATH) if os.path.exists(RANDOM_DATA_PATH+file+'/TLE_density_all.mat')]
    working_datasets = []
    for ds in all_datasets:
        try:
            mat = h5py.File(f'{RANDOM_DATA_PATH}/{ds}/TLE_density_all.mat' , 'r')
            working_datasets.append(ds)
        except:
            pass
    
    print("Number of datasets that can be opened is: ", len(working_datasets), " out of ", len(all_datasets))

    datasets = random.sample(working_datasets, args.train_size) if len(working_datasets) > args.train_size else working_datasets

    train_params, train_dls, train_Xs = [], [], []
    for ds_name in datasets: 
        with open(f"{RANDOM_DATA_PATH}{ds_name}/parameters.txt", "r") as file:  
            content = file.read().split()[4:]
            params = [float(content[i]) for i in [0,2,4,6]]
            train_params.append(params)

        X, X_sw = get_train_data(args, ds_name, params)

        ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)

        samples_per_simulation = X_sw.shape[0]//(X.shape[0])
        train_tl = TfmdLists(calculate_sample_idxs(range(X.shape[0]), samples_per_simulation), DensityTupleTransform(ds))
        train_dls.append(train_tl)
        train_Xs.append(X)

        print("Loaded dataset: ", [round(i, 3) for i in params], " Shapes: ", X.shape, X_sw.shape)


    valid_dls, valid_Xs = [], []
    parameters = [[ip, lr, pmd, cam] for ip in [1, 2] for lr in [1, 2] for pmd in [0.95, 0.97, 0.99] for cam in [0.95, 0.97, 0.99]]
    for params in parameters:
        X, X_sw = get_valid_data(args, params)

        ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
        samples_per_simulation = X_sw.shape[0]//(X.shape[0])
        val_tl = TfmdLists(calculate_sample_idxs(range(X.shape[0]), samples_per_simulation), DensityTupleTransform(ds))
        valid_dls.append(val_tl)
        valid_Xs.append(X)

        print("Loaded dataset: ", params, " Shapes: ", X.shape, X_sw.shape)


    train = np.concatenate(train_Xs, axis=0)   
    mocat_stats = (np.mean(train), np.std(train))

    train, valid = ConcatDataset(train_dls), ConcatDataset(valid_dls)
    dls = DataLoaders.from_dsets(train, valid, bs=config.bs, device=default_device(),
                        after_batch=[Normalize.from_stats(*mocat_stats)] if \
                        config.norm else None,
                        num_workers=config.num_workers)

    return dls, train_params



def train(config):
    # data setup
    dls, train_params  = get_dls_random(config)
    loss_func, metrics = get_loss_function(config), get_metrics(config)


    # model setup
    config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
    model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
    learn = Learner(dls, model, loss_func=loss_func, cbs=[], metrics=metrics)
    lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()


    # Training
    print("MODEL SIZE: ", get_n_params(learn), "\n")
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    print("Training DONE!")


    # Directories to save results 
    date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now())
    os.makedirs(f"results/{date}")
    os.makedirs(f"models/{date}")

    # Save training history
    save_training_metrics(learn, metrics, save_folder=f'results/{date}/')

    # Save config as txtfile
    with open(f"results/{date}/config.txt", 'w') as f:
        json.dump(config, f, indent=4)

    # Save model
    if config.save_model:
        torch.save(learn.model.state_dict(), f"results/{date}/model.pth")

        ts, g, d, c, h, n = config.train_size, config.guidance, config.dual, config.channel, config.horizon, config.n_epoch
        torch.save(learn.model.state_dict(), f"models/{date}/train_size_{ts}_guidance_{g}_dual_{d}_channel_{c}_horizon_{h}_n_epochs_{n}.pth")


    # Evalution
    model.eval()

    dual = 2 if config.dual else 1
    parameters = [[ip, lr, pmd, cam] for ip in [1, 2] for lr in [1, 2] for pmd in [0.95, 0.97, 0.99] for cam in [0.95, 0.97, 0.99]]


    for params in parameters:
        val = get_valid_data(config, params)[0].mean(axis=0)  
        val = torch.tensor(val).float()

        # Iterative forecasting
        initial_seq = val[:config.lookback].to(default_device())
        inp = tuple(i.unsqueeze(0) for i in initial_seq)
        preds = [initial_seq[:,:dual,:,:]]
        
        n_iter = N//config.lookback
        for iter in range(1, n_iter+1):
            p = model(inp)
            preds.append(torch.vstack(p))

            if iter==n_iter:
                break

            if config.guidance:
                target = val[iter*config.horizon:(iter+1)*config.horizon].to(default_device())
                guidance = target[:,dual:,:,:] 
                inp = tuple(torch.cat([p[i], guidance[i:i+1]], axis=1)  for i in range(config.lookback))
            else:
                inp = p

        preds = torch.vstack(preds)[:N].cpu().detach()

        ds = 'x' + 'x'.join([str(i) for i in params])
        if config.guidance:
            val = val[:,:dual,:,:]



if __name__ == "__main__":    

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    parser.add_argument("--train_size", type = int, default = 10) # Number of training datasets to use for training
    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--lookback", type = int, default = 50)
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 100) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 150)
    parser.add_argument("--n_epoch", type = int, default = 30)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--log", type = int, default = 1)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    parser.add_argument("--norm", type = int, default = 1) # whether or not to normalize (Z-norm) during training
    parser.add_argument("--guidance", type = int, default = 0)  # whether or not to use guidance (parameters added along image channel)
    parser.add_argument("--save_model", type = int, default = 0) # whether or not to save the model

    parser.add_argument("--eval", type = int, default = 0) # whether or not to evaluate model or train model
    parser.add_argument("--dual", type = int, default = 0) # whether to train on 2 channels or not (inactive and active objects)
    parser.add_argument("--channel", type = int, default = 0) # which channel to train on if dual is 0, 0 for inactive, 1 for active 2 for sum of the two channels   


    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)


    # Settings
    model_type = 'convgru'
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)

    if args.guidance == 1:
        config_base['convgru']['n_in'] = 6 if args.dual == 1 else 5
    else:
        config_base['convgru']['n_in'] = 2 if args.dual == 1 else 1

    config_base['convgru']['n_out'] = 2 if args.dual == 1 else 1


    config = AttrDict(config_base)
    config.partial_loss = [0] if args.partial_loss == 1 else None

    keys = ['train_size', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'log', 'average', 'norm', 'save_model', 'guidance', 'dual', 'eval', 'channel']  
    for key in keys:
        config[key] = arg_dict[key]


    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    if args.eval:
        eval(config)
    else:
        train(config)  

