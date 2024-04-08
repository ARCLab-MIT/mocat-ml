import torch, argparse
from vit_pytorch import ViT

import sys
sys.path.append('..')
from fastai.vision.all import *
from mocatml.utils import *
convert_uuids_to_indices()
from mocatml.data import *
from mocatml.models.conv_rnn import *

from tsai.utils import yaml2dict, dict2attrdict
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        return self.data[i], self.data[i]
    
    
class Decoder(torch.nn.Module):
    def __init__(self, enc_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(enc_dim, 36*99)
        
    def forward(self, inp):
        return self.linear(inp)
    
    
class AutoEncoder(torch.nn.Module):
    def __init__(self, enc_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = ViT(
                            image_size = (36, 99),
                            patch_size = 9,
                            num_classes = enc_dim,
                            dim = 768,
                            depth = 12,
                            heads = 12,
                            mlp_dim = 3072,
                            dropout = 0.1,
                            emb_dropout = 0.1, 
                            channels = 1
                            )
        
        self.decoder = Decoder(enc_dim)
        
    def forward(self, inp):
        inp = inp.to(DEVICE, non_blocking = True)
        bs = inp.shape[0]
        enc = self.encoder(inp)
        return self.decoder(enc).reshape(bs, 1, 36, 99)
    
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description = "")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "x2x2", help = "datasets to train on") #1 means on all datasets otherwise list the dataset like this 'x2x2, x5x5, x10x10'
    
    parser.add_argument("--n_epoch", type = int, default = 20)
    
    parser.add_argument("--split", type = float, default = 0.95)
    
    parser.add_argument("--bs", type = int, default = 512)
    
    parser.add_argument("--enc_dim", type = int, default = 128, help = "latent vector size")

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    config_base = yaml2dict('./config/data-gen.yaml', attrdict=True)
    config = AttrDict(config_base)
        
    print("CONFIG \n", json.dumps(config, indent=4))
    config_base = yaml2dict('./config/base.yaml', attrdict=True)
    config_base.convgru = yaml2dict('./config/convgru/convgru.yaml', attrdict=True)
    config = AttrDict(config_base)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE: ", DEVICE)

    data = np.load(Path(config.data.path + f'{args.dataset}.npy').expanduser(), 
                   mmap_mode='c' if config.mmap else None)
    
    data = data[:, :None][:,::config.stride,:].reshape(-1, 1, *(data.shape[2:]))
    data = torch.tensor(data, dtype=torch.float32)

    indices = [i for i in range(data.shape[0])]
    random.shuffle(indices)

    x = int(len(indices)*args.split)
    train_indices, val_indices = indices[:x], indices[x:]
    print(len(train_indices), len(val_indices))

    train_ds = CustomDataset(data[train_indices])
    val_ds = CustomDataset(data[val_indices])

    AE = AutoEncoder(args.enc_dim).to(DEVICE)
    dls = DataLoaders.from_dsets(train_ds, val_ds, bs = args.bs, pin_memory=True, shuffle = True)
    learn = Learner(dls, AE, loss_func = F.mse_loss, cbs = [ShowGraphCallback()])
    print(f'Model size {AE.size()}')
    lr = learn.lr_find().valley
    learn.fit_one_cycle(args.n_epoch)


