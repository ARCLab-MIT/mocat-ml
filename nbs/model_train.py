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

my_setup()

def train_on_dataset(ds_list, config):
    # only implemented for convgru (add more architectures)
    dls, splits, Xs  = get_dataloader_from_dslist(ds_list, config)
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
    print("Training DONE!")

    date, c = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now()), config
    save_to = f"results/convgru/{date}_l{c.lookback}_s{c.stride}_ds_{c.ds_list}_loss_{c.loss}_bs{c.bs}_sample_{c.sample}"
    save_path = f"{save_to}/long_term_prediction_after_{c.n_epoch}_epochs"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(8, 5))  
    learn.recorder.plot_loss(skip_start=0, with_valid=True)
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.title('Training and Validation loss')
    plt.savefig(save_to + f"/training_val_loss.png")
    plt.close()

    for ds_name, X, split in zip(ds_list, Xs, splits):
        do_long_term_prediction(learn, config, ds_name, X, split, save_path)
    return learn 


def do_long_term_prediction(learn, config, ds_name, X, split, save_path):
    #TODO implemented for only case when gap is zero
    
    # Getting random simulation run from validation set
    idx = random.choice(split[1])
    n_iter = 2436//(config.lookback  + config.gap) - 1

    ds = DensityData(X[idx:idx+1], lbk=config.lookback, h=config.horizon, gap=config.gap)
    tl = TfmdLists(range(len(ds)), DensityTupleTransform(ds))
    dl = TfmdDL(tl, bs=learn.dls.valid.bs)

    inp, p, t = learn.get_preds(dl=dl, with_input=True)
    ds = dl.ds
    losses = [learn.loss_func(p,t).item()]
    predictions = [torch.vstack(inp+p)]
    for iter in range(n_iter-1):
        data_copy = ds.data[:,(iter+1)*(ds.lbk+ds.gap):\
                                 (iter+1)*(ds.lbk+ds.gap) + ds.lbk + ds.h].copy()
        
        ds_copy = DensityData(data_copy, lbk=ds.lbk, h=ds.h, gap=ds.gap)
        tl = TfmdLists(range(len(ds_copy)), DensityTupleTransform(ds_copy))
        # Save the targets before replacing data
        t = stack_density_list_as_preds_targs([y for _,y in tl])
        # Replace the first inputs of the dataset with the predictions
        p_dseqs = [DensitySeq.from_preds_or_targs(p, i, to_array=True) \
                   for i in range(len(p[0]))]
        
        preds_data = np.stack(p_dseqs).squeeze()
        ds_copy.data[:,:ds_copy.lbk] = preds_data
        dl_new = dl.new(TfmdLists(range(len(ds_copy)), 
                                  DensityTupleTransform(ds_copy)))
        p,_ = learn.get_preds(dl=dl_new, with_input=False)
        predictions.append(torch.vstack(p))
        losses.append(learn.loss_func(p,t).item())
    
    predictions = torch.squeeze(torch.vstack(predictions), dim=1)
    target_pred = torch.cat((torch.tensor(X[idx][:min(predictions.shape[0], 2436)]), predictions), 2)

    vmin, vmax = torch.min(target_pred), torch.max(target_pred)
    make_and_save_gif(target_pred, f"{save_path}/{ds_name}_predictions.gif", vmin, vmax, n_iter)

    plt.figure(figsize=(8, 5))  
    plt.plot(losses, label='Loss', color='blue')
    plt.xlabel('Forecast Iteration')
    plt.ylabel(f'Loss ({config.loss})')
    plt.title('Loss Evolution During Long Term Prediction')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(save_path + f"/{ds_name}_losses.png")
    plt.close()

    return learn


if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--ds_list", nargs='+', default = ["x3x3", "x4x4", "x5x5", "x6x6", "x8x8"], help = "datasets to train on")  
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
    parser.add_argument("--log", type = int, default = 0)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 0) # whether or not to use averaging 
    
    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds_list', 'horizon', 'lookback', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'sample']:
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss

    print("CONFIG \n", json.dumps(config, indent=4))

    # # Training
    print(args.ds_list)
    learn = train_on_dataset(args.ds_list, config)

