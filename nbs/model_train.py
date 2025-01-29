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
import wandb, json, argparse, os, torch, random, datetime, math
import numpy as np

from loss_functions import *
from utils import *
from diffusion_utils import  make_and_save_gif

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
    save_to = f"results/convgru/init_pop_{c.init_pop}_launch_rate_{c.launch_rate}/{date}_l{c.lookback}_gap_{c.gap}_s{c.stride}_loss_{c.loss}_metric_{c.metric}_bs{c.bs}_sample_{c.sample}_log_{c.log}_average_{c.average}"
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

    metric_scores_by_year, metric_scores_full = {}, {}
    for ds_name, X, split in zip(ds_list, Xs, splits):
        score_year, score_full = do_long_term_prediction(learn, config, ds_name, X, split, save_path)
        metric_scores_by_year[ds_name] = score_year  
        metric_scores_full[ds_name] = score_full

    plot_metrics(metric_scores_by_year, metric_scores_full, save_to+'/')
    return learn 


def do_long_term_prediction(learn, config, ds_name, X, split, save_path): # only gap <= 0
    if config.gap > 0:
        print("Only implemented for gap = 0")
        return

    # Getting random simulation run from validation set
    idx = random.choice(split[1])
    ds = DensityData(X[idx:idx+1], lbk=config.lookback, h=config.horizon, gap=config.gap)
    tl = TfmdLists(range(len(ds)), DensityTupleTransform(ds))
    dl = TfmdDL(tl, bs=learn.dls.valid.bs)

    inp, p, t = learn.get_preds(dl=dl, with_input=True)
    ds = dl.ds
    losses = [learn.loss_func(p,t).item()]
    predictions = [torch.vstack(inp+p)]

    n_iter = 2436//(config.lookback  + config.gap) - math.ceil(config.lookback/(config.lookback + config.gap))
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

        # Save the predictions
        predictions.append(torch.vstack(p[-ds.gap:]))
        losses.append(learn.loss_func(p,t).item())
    
    predictions = torch.squeeze(torch.vstack(predictions), dim=1)
    n = min(2436, len(predictions))
    target, predictions = torch.tensor(X[idx][:n]), predictions[:n]

    if config.log:
        target, predictions = torch.exp(target)-1, torch.exp(predictions)-1

    vmin, vmax = min([torch.min(target).item(), torch.min(predictions).item()]), max([torch.max(target).item(), torch.max(predictions).item()])
    make_and_save_gif(target, predictions, config, f"{save_path}/{ds_name}_predictions.gif", vmin, vmax, n_iter)

    plt.figure(figsize=(8, 5))  
    plt.plot(losses, label='Loss', color='blue')
    plt.xlabel('Forecast Iteration')
    plt.ylabel(f'Loss ({config.loss})')
    plt.title('Loss Evolution During Long Term Prediction')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(save_path + f"/{ds_name}_losses.png")
    plt.close()

     # plotting metric results
    preds = [predictions[int((n-1)/10*i)] for i in range(1, 11)]
    targets = [target[int((n-1)/10*i)] for i in range(1, 11)]

    scores_by_year = [SMAPELoss()(pred, tar).item() for pred, tar in zip(preds, targets)]
    scores_full = [SMAPELoss()(pred, tar).item() for pred, tar in zip(predictions, target)]
    return scores_by_year, scores_full


if __name__ == "__main__":    
    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--launch_rate", nargs="+", type=int, default = [2, 5])  # range of launch rates to use for training 
    parser.add_argument("--init_pop", nargs="+", type=int, default = [2, 5])   # range of initial populations to use for training
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    parser.add_argument("--horizon", type = int, default = 50) 
    parser.add_argument("--lookback", type = int, default = 50)
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 100) 
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    parser.add_argument("--bs", type = int, default = 16)
    parser.add_argument("--n_epoch", type = int, default = 30)
    parser.add_argument("--sel_steps", type = int, default = None)
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--loss", type = str, default = 'mae')
    parser.add_argument("--sample", type = int, default = 1) # whether or not to sample to 32x32
    parser.add_argument("--log", type = int, default = 0)  # whether or not to train on log(N)
    parser.add_argument("--average", type = int, default = 1) # whether or not to use averaging 
    parser.add_argument("--normalize", type = int, default = 0) # whether or not to normalize from -1 to 1
    parser.add_argument("--metric", type = str, default = 'smape') 

    # Set defaults 
    args = parser.parse_args()
    args.init_pop = [i for i in range(args.init_pop[0], args.init_pop[1]+1)]    
    args.launch_rate = [i for i in range(args.launch_rate[0], args.launch_rate[1]+1)]   
    args.ds_list = [f'x{init_pop}x{launch_rate}' for init_pop in args.init_pop for launch_rate in args.launch_rate] # Default datasets to train on
    arg_dict = vars(args)

    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base[model_type] = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    for key in ['ds_list', 'launch_rate', 'init_pop', 'horizon', 'lookback', 'gap', 'stride', 'bs', 'n_epoch', 'sel_steps', 'key', 'loss', 'sample', 'log', 'average', 'metric']:   
        config[key] = arg_dict[key]

    if config['loss'] == 'mbd':
        config['alpha'] = 0.5 #set alpha here for mbd loss
    
    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    learn = train_on_dataset(args.ds_list, config)    
