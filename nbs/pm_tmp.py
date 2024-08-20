# -*- coding: utf-8 -*-
"""pm_tmp.ipynb

Automatically generated.

Original file is located at:
    /home/gridsan/vrodriguez/Github/mocat-ml/nbs/pm_tmp.ipynb
"""

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
import wandb

from fastai.callback.schedule import LRFinder

@patch_to(LRFinder)
def after_fit(self):
    self.learn.opt.zero_grad() # Needed before detaching the optimizer for future fits
    tmp_f = self.path/self.model_dir/self.tmp_p/'_tmp.pth'
    if tmp_f.exists():
        self.learn.load(f'{self.tmp_p}/_tmp', with_opt=True, device='cpu')
        self.tmp_d.cleanup()

my_setup()

config_base = yaml2dict('./config/base.yaml', attrdict=True)
config_base.convgru = yaml2dict('./config/convgru/convgru.yaml', attrdict=True)
#config = AttrDict({**config_base, **config_e2e})
config = AttrDict(config_base)
config

# Parameters
config.device = "cuda"
config.n_epoch = 20
config.wandb.enabled = True
config.wandb.log_learner = False
config.lookback = 4
config.horizon = 4
config.gap = -3
config.stride = 4
config.bs = 32
config.save_learner = True
config.lr_max = 0.001

# Set device
default_device(0 if config.device == 'cpu' else config.device)

run = wandb.init(dir=ifnone(config.wandb.dir, '../'),
                 project=config.wandb.project, 
                 config=config,
                 group=config.wandb.group,
                 mode=config.wandb.mode, 
                 anonymous='never') if config.wandb.enabled else None
config = dict2attrdict(run.config) if config.wandb.enabled else config
print(config)

data = np.load(Path(config.data.path).expanduser(), 
               mmap_mode='c' if config.mmap else None)
data = data[:, :config.sel_steps]
data.shape

data_sw = np.lib.stride_tricks.sliding_window_view(data, 
                                               config.lookback + config.horizon + config.gap, 
                                               axis=1)[:,::config.stride,:]
samples_per_simulation = data_sw.shape[1]
data_sw = data_sw.transpose(0,1,4,2,3)
data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
data_sw.shape

# Split by simulation
splits = RandomSplitter()(data)
splits

ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)
train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)
len(train_idxs), len(valid_idxs)

mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))
mocat_stats

# Create dataloaders
train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config.bs, device=default_device(),
                            after_batch=[Normalize.from_stats(*mocat_stats)] if \
                             config.normalize else None,
                            num_workers=config.num_workers)
#dls.show_batch()
foo, bar = dls.one_batch()
len(foo), len(bar)

if config.partial_loss is not None:
    loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
    full_loss = StackLoss()
    full_loss.__name__ = "full_loss"
    metrics = [full_loss] # [StackLoss()]
else:
    loss_func = StackLoss(MSELossFlat())
    metrics = []

config.convgru.norm = NormType.Batch if config.convgru.norm == 'batch' else None
model = StackUnstack(SimpleModel(**config.convgru)).to(default_device())
wandbc = WandbCallback(log_preds=False, log_model=False) if config.wandb.enabled else None
cbs = L() + wandbc
learn = Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics)
learn.splits = splits # This is needed for the evaluation notebook
lr_max = config.lr_max if config.lr_max is not None else learn.lr_find()

learn.fit_one_cycle(config.n_epoch, lr_max=lr_max)

p,t = learn.get_preds()
len(p), p[0].shape

def show_res(t, idx, figsize=(8,4)):
    density_seq = DensitySeq.create([t[i][idx] for i in range(len(t))])
    density_seq.show(figsize=figsize);

# Log inference time
k = random.randint(0, dls.valid.n)
foo = TfmdLists([k], DensityTupleTransform(ds))
bar = dls.valid.new(foo)
start_time = time.time()
learn.get_preds(dl=bar)
wandb.log({'inference_time': time.time() - start_time})

# Remove the wandb callback to avoid errors when downloading the learner
if config.wandb.enabled:
    learn.remove_cb(wandbc)

if config.save_learner:
    learn.model_dir = config.tmp_folder
    learn.export(f'{config.tmp_folder}/learner.pkl', pickle_protocol=4)

# Save locally and in wandb if it's enabled
if run is not None and config.wandb.log_learner:
    # Save the learner (all tmp/dls, tmp/model.pth, and tmp/learner.pkl). 
    run.log_artifact(config.tmp_folder, type='learner', name='density-forecaster')

if run is not None:
    run.finish()