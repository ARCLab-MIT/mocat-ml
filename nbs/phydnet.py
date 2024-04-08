import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.conv_rnn import *
from mocatml.models.phy_original import *
from mocatml.models.transformer import *
from mocatml.models.seq2seq import TeacherForcing
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
        
        
        
        

config_base = yaml2dict('./config/base.yaml', attrdict=True)
config_base.phdnet = yaml2dict('./config/phdnet/phydnet.yaml', attrdict=True)
#config = AttrDict({**config_base, **config_e2e})
config = AttrDict(config_base)


run = wandb.init(dir=ifnone(config.wandb.dir, '../'),
                 project=config.wandb.project, 
                 config=config,
                 group=config.wandb.group,
                 mode=config.wandb.mode, 
                 anonymous='never') if config.wandb.enabled else None
config = dict2attrdict(run.config) if config.wandb.enabled else config


data = np.load(Path(config.data.path + config.data.dataset[0] + '.npy').expanduser(), 
               mmap_mode='c' if config.mmap else None)

data = data[:, :config.sel_steps]

data_sw = np.lib.stride_tricks.sliding_window_view(data, 
                                               config.lookback + config.horizon + config.gap, 
                                               axis=1)[:,::config.stride,:]
samples_per_simulation = data_sw.shape[1]
data_sw = data_sw.transpose(0,1,4,2,3)
data_sw = data_sw.reshape(-1, *data_sw.shape[2:])
data_sw = data_sw[:, :, :32, :32]



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
    
# #export
# class PHyCallback(Callback):
#     def after_pred(self):
#         self.learn.pred, self.loss_phy = self.pred
#     def after_loss(self):
#         self.learn.loss += self.loss_phy
        
# mse_loss = StackLoss(MSELossFlat(axis=1))
# metrics = []


# phycell =  PhyCell(input_shape=(16, 16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7)) 
# convlstm = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3))   
# encoder =  EncoderRNN(phycell, convlstm)

# model = StackUnstack(PhyDNet(encoder, sigmoid=False, moment=True), dim=1).cuda()
# cbs = L() + [ShowGraphCallback()] + [TeacherForcing(10), PHyCallback()]
# learn = Learner(dls, model, loss_func=mse_loss, cbs=cbs, metrics=metrics, opt_func = ranger)
# lr_max = learn.lr_find()

# learn.fit_flat_cos(config.n_epoch, 1e-4)
# print("FINISHEDDDDDD")


loss_func = StackLoss(CrossEntropyLossFlat(axis=2))

model = StackUnstack(TransformerTS(n_in=1, n_out=2))
print(model)