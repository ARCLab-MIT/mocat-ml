import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.conv_rnn import *
from mocatml.models.transformer import *
from mocatml.models.phy_original import *
from mocatml.models.seq2seq import TeacherForcing
from mygrad import sliding_window_view
from tsai.imports import my_setup
from tsai.utils import yaml2dict, dict2attrdict
from fastai.callback.schedule import valley, steep
from fastai.callback.wandb import WandbCallback
import wandb

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(torch.cuda.get_device_name())


from fastai.callback.schedule import LRFinder

@patch_to(LRFinder)
def after_fit(self):
    self.learn.opt.zero_grad() # Needed before detaching the optimizer for future fits
    tmp_f = self.path/self.model_dir/self.tmp_p/'_tmp.pth'
    if tmp_f.exists():
        self.learn.load(f'{self.tmp_p}/_tmp', with_opt=True, device='cpu')
        self.tmp_d.cleanup()

config_base = yaml2dict('./config/base.yaml', attrdict=True)
config = AttrDict(config_base)

data = np.load(Path(config.data.path + config.data.dataset[0] + '.npy').expanduser(), 
               mmap_mode='c' if config.mmap else None)

data = data[:, :config.sel_steps]

data_sw = np.lib.stride_tricks.sliding_window_view(data, 
                                               config.lookback + config.horizon + config.gap, 
                                               axis=1)[:,::config.stride,:]
samples_per_simulation = data_sw.shape[1]
data_sw = data_sw.transpose(0,1,4,2,3)
data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
l, h, width, height = data_sw.shape
data_sw = data_sw.reshape(l, h, 3, width, height//3)
data_sw = data_sw[:, :, :, :32, :32]
print(data_sw.shape)

splits = RandomSplitter()(data)
ds = DensityData(data_sw, lbk=config.lookback, h=config.horizon, gap=config.gap)


train_idxs = calculate_sample_idxs(splits[0], samples_per_simulation)
valid_idxs = calculate_sample_idxs(splits[1], samples_per_simulation)

mocat_stats = (np.mean(data[splits[0]]), np.std(data[splits[0]]))

train_tl = TfmdLists(train_idxs, DensityTupleTransform(ds))
valid_tl = TfmdLists(valid_idxs, DensityTupleTransform(ds))
dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=config.bs, device=default_device(),
                    after_batch=[Normalize.from_stats(*mocat_stats)] if \
                    config.normalize else None,
                    num_workers=config.num_workers)
                    
config.partial_loss = [0]
if config.partial_loss is not None:
    loss_func = PartialStackLoss(config.partial_loss, loss_func=MSELossFlat())
    full_loss = StackLoss()
    full_loss.__name__ = "full_loss"
    metrics = [full_loss] # [StackLoss()]
else:
    loss_func = StackLoss(MSELossFlat())
    metrics = []
    
loss_func = StackLoss(MSELossFlat())
metrics = []

model = StackUnstack(TransformerTS(n_in=3, n_out=3))
cbs = L() + [ShowGraphCallback()]
learn = Learner(dls, model, loss_func=loss_func, metrics=metrics, splitter=partial(tf_split, stacked=True), cbs=cbs).to_fp16()
lr_max = learn.lr_find()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print("model size: ", get_n_params(learn))
learn.fit_one_cycle(50, 1e-4)
