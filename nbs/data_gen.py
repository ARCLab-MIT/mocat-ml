import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.conv_rnn import *
from mygrad import sliding_window_view
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from fastai.callback.wandb import WandbCallback
import wandb, json, argparse

my_setup()

def train_on_dataset(model_type, dataset, config):
    # only implemented for convgru (add more architectures)
    
    dls, splits = get_dataloader(dataset, config)
    
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
    learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)
    
    # evaluation on datasets
    if config.data.include:
        name = f'learner_trained_on_{dataset}'
    else:
        dataset_to_train_on = config.data.dataset.copy()
        dataset_to_train_on.remove(dataset)
        name = f'learner_trained_on_{dataset_to_train_on}'
        
    eval_data = {}
    eval_data[name] = {__:[] for __ in config.data.dataset}
    
    for ds_name in config.data.dataset:
        X = np.load(Path(config.data.path + ds_name + '.npy').expanduser(), mmap_mode='c' 
                    if config.mmap else None)

        X_sw = np.lib.stride_tricks.sliding_window_view(X[:, :config.sel_steps], 
                                       config.lookback + config.horizon, 
                                       axis=1)[:,::config.stride,:]
        samples_per_simulation = X_sw.shape[1]
        X_sw = X_sw.transpose(0,1,4,2,3)
        X_sw = X_sw.reshape(-1, *X_sw.shape[2:])

        train_stats = (learn.dls.train.after_batch.mean, learn.dls.train.after_batch.std)
        ds = DensityData(X_sw, lbk=config.lookback, h=config.horizon)
        tl = TfmdLists(range(len(ds)), DensityTupleTransform(ds))
        dl = TfmdDL(tl, bs=learn.dls.valid.bs, shuffle=False, num_workers=0, 
                    after_batch=Normalize.from_stats(*train_stats))

        inps, preds, targs = learn.get_preds(dl=dl, with_input=True)
        valid_loss = learn.loss_func(preds, targs)

        eval_data[name][ds_name] = valid_loss.item()
        
    return name, eval_data


def get_dataset(dataset, config):
    data = np.load(Path(config.data.path + dataset + '.npy').expanduser(), 
               mmap_mode='c' if config.mmap else None)

    data = data[:, :config.sel_steps]
    data_sw = np.lib.stride_tricks.sliding_window_view(data, config.lookback + config.horizon + config.gap, axis=1)[:,::config.stride,:]
    samples_per_simulation = data_sw.shape[1]
    data_sw = data_sw.transpose(0,1,4,2,3)
    data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
    return data, data_sw



def get_dataloader(dataset, config):
    if config.data.include:
        data, data_sw = get_dataset(dataset, config)
        
    else:
        datasets_to_train = config.data.dataset.copy()
        datasets_to_train.remove(dataset)
        datasets = []
        for ds in datasets_to_train:
            if ds == dataset: pass
            datasets.append(get_dataset(ds, config))

        data = np.concatenate([i[0] for i in datasets])
        data_sw = np.concatenate([i[1] for i in datasets])
        
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
    
    return dls, splits

if __name__ == "__main__":
    
    # TODO - add wandb implementation and more architectures

    # Parser
    parser = argparse.ArgumentParser(description = "Checking how model generalizes")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "1", help = "datasets to train on") #1 means on all datasets otherwise list the dataset like this 'x2x2, x5x5, x10x10'
    
    parser.add_argument("--model", type = str, default = "convgru", help = "architecture to use")
    
    parser.add_argument("--include", type = int, default = 1) # 1 - include, 0 -  exclude
    
    parser.add_argument("--horizon", type = int, default = 4)
    
    parser.add_argument("--lookback", type = int, default = 4)
    
    parser.add_argument("--stride", type = int, default = 8)
    
    parser.add_argument("--partial_loss", type = int, default = 0) #1 - True 0 - False
    
    parser.add_argument("--bs", type = int, default = 32)
    
    parser.add_argument("--num_run", type = int, default = 5)
    
    parser.add_argument("--n_epoch", type = int, default = 20)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    model_type = args.model
    config_base = yaml2dict('./config/data-gen.yaml', attrdict=True)
    config_base.convgru = yaml2dict(f'./config/{model_type}/{model_type}.yaml', attrdict=True)
    config = AttrDict(config_base)
    
    config.partial_loss = [0] if args.partial_loss == 1 else None
    config.data.include = True if args.include else False
    config.data.to_run = 1 if args.dataset == '1' else args.dataset.split(',')
    for key in ['horizon', 'lookback', 'stride', 'bs', 'num_run', 'n_epoch']:
        config[key] = arg_dict[key]

    to_run = config.data.dataset if config.data.to_run == 1 else config.data.to_run
    
    print("CONFIG \n", json.dumps(config, indent=4))

    result_dict = {}
    result_dict['model'] = model_type
    result_dict['config'] = config
    result_dict['result'] = {}
    path = f"result/horizon_{config.horizon}_include_{config.data.include}_partial_{args.partial_loss}"
    
    for dataset in to_run:
        for _ in range(config.num_run):

            model_name, result = train_on_dataset(model_type, dataset, config)
            if model_name not in result_dict['result']:
                result_dict['result'][model_name] = {i: [] for i in config.data.dataset}
                
            for key in result[model_name]:
                result_dict['result'][model_name][key].append(result[model_name][key])
                    
            print(f"Finished run number {_} \nDataset: {dataset}")
            
    with open(f"{path}.json", "w") as outfile:
        json.dump(result_dict, outfile)
    
    print("FINISHEDDD")
            