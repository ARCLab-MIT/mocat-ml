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
import wandb, json, argparse, os

my_setup()

def train_on_dataset(model_type, dataset, config):
    # only implemented for convgru (add more architectures)
    
    dls, splits, X, X_sw = get_dataloader(dataset, config)
    
    if config.partial_loss is not None:
        loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
        full_loss = StackLoss()
        full_loss.__name__ = "full_loss"
        metrics = [full_loss] # [StackLoss()]
    else:
        loss_func = StackLoss(MSELossFlat())
        metrics = []

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
    # learn.fit(config.n_epoch, 1e-1)
    # learn.eval()
    # plot_preds(learn, config, X, X_sw, dataset)
    return learn

def get_dataset(dataset, config):
    if dataset == 'comb':
        data = np.load(Path('~/mocat-ml/data/comb_am_rp.npy').expanduser(), 
               mmap_mode='c' if config.mmap else None)
    else:
        data = np.load(Path(config.data.path + dataset + '.npy').expanduser(), 
                mmap_mode='c' if config.mmap else None)

    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    samples_per_simulation = data_sw.shape[1]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw

def get_dataloader(dataset, config):
    data, data_sw = get_dataset(dataset, config)
    print(data_sw.shape, data.shape)
    
    splits = RandomSplitter()(data)
    ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
    samples_per_simulation = data_sw.shape[1]
    train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
    valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

    mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))

    train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
    valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
    dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config.bs, device=default_device(),
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


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def plot_preds(learn, config, X, X_sw, dataset_name):
    train_stats = (learn.dls.train.after_batch.mean, learn.dls.train.after_batch.std)
    ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon)
    tl = TfmdLists(range(len(ds)), DensityTupleTransform(ds))
    dl = TfmdDL(tl, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
            after_batch=Normalize.from_stats(*train_stats))
    inps, _, _ = learn.get_preds(dl=dl, with_input=True)

    ds_full = DensityData(X, lbk=config.lookback, h=config.horizon)
    tl_full = TfmdLists(range(len(ds_full)), DensityTupleTransform(ds_full))
    dl_full = TfmdDL(tl_full, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
                after_batch=Normalize.from_stats(*train_stats))

    xb,yb = dl_full.one_batch()

    n_iter = X.shape[1]//config.horizon - 1
    n_iter_half = n_iter//2

    preds,targs,losses = learn.get_preds_iterative(dl=dl_full, n_iter=n_iter, track_losses=True)
    save_path = f"plots/{dataset_name}/stride_{config.stride}_bs_{config.bs}/num_epochs_{config.n_epoch}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    learn.show_preds_at(0, p=preds, t=targs, inp=inps, save=True, save_path = save_path, with_targets=True, 
                    with_input=True, start_epoch=(n_iter-1)*config.horizon,
                   titles=["Input", "100 year-ahead predictions with non-overlapping model", 
                           "100 year-ahead targets"])

                        
if __name__ == "__main__":
    
    # TODO - add wandb implementation and more architectures

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "x2x2", help = "dataset to train on")     
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

    print("CONFIG \n", json.dumps(config, indent=4))

    # Training
    learn = train_on_dataset(model_type, args.dataset, config)

    
    # Loss plot
    path = f'plots/{args.dataset}/stride_{config.stride}_bs_{config.bs}/num_epochs_{config.n_epoch}/'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in [0, 0.5, 0.75, 0.9]:
        num_epochs_toshow = config.n_epoch - int(i*config.n_epoch)
        fig, ax = plt.subplots()
        skip_start = int(len(learn.recorder.losses) * i)
        plot_loss(learn.recorder, skip_start=skip_start, ax=ax)
        ax.set_title('learning curve full' if skip_start == 0 else f'learning curve last {num_epochs_toshow} epochs')
        name = 'full' if i==0 else f'last {num_epochs_toshow} epochs'
        plt.savefig(f'{path}{name}.png')
        plt.show()
