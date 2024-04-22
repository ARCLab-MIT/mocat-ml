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

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(2,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class CustomDataset(Dataset):
    def __init__(self, data, horizon = 0):
        self.data = data
        self.horizon = horizon

    def __len__(self):
        return self.data.shape[0] - self.horizon
    
    def __getitem__(self, i):
        if self.horizon < 2:
            return self.data[i], self.data[i+self.horizon]
        return self.data[i:i+self.horizon].squeeze(), self.data[i+self.horizon]


class Decoderr(torch.nn.Module):
    def __init__(self, enc_dim, horizon, debug = False):
        super(Decoderr, self).__init__()
        self.d1 = dcgan_upconv(horizon, 2*horizon, stride=1)
        self.d2 = dcgan_upconv(2*horizon, 4*horizon, stride=1)
        self.d3 = dcgan_upconv(4*horizon, 6*horizon, stride=1)
        self.linear = nn.Linear(6*enc_dim*horizon, 36*99)
        self.horizon = horizon
        self.debug = debug

    def forward(self, inp):
        bs, n = inp.shape[0], inp.shape[1]  
        inp = inp.reshape(bs, n, 16, -1)
        if self.debug: print(inp.shape)
        d1 = self.d1(inp)
        if self.debug: print(d1.shape)
        d2 = self.d2(d1)
        if self.debug: print(d2.shape)
        d3 = self.d3(d2)
        if self.debug: print(d3.shape)
        d3 = d3.reshape(bs, 1, 6*self.horizon*d3.shape[2]*d3.shape[3])
        return self.linear(d3)


class AutoEncoderr(torch.nn.Module):
    def __init__(self, enc_dim, horizon = 1, act = False):
        super(AutoEncoderr, self).__init__()
        self.encoder = ViT(
                            image_size = (36, 99),
                            patch_size = 9,
                            num_classes = enc_dim,
                            dim = 256,
                            depth = 3,
                            heads = 3,
                            mlp_dim = 1024,
                            dropout = 0.1,
                            emb_dropout = 0.1, 
                            channels = 1
                            )

        self.decoder = Decoderr(enc_dim, horizon)
        self.horizon = horizon
        self.act = nn.ReLU() if act else None

    def forward(self, inp):
        inp = inp.to(DEVICE, non_blocking = True)
        encoded = []
        for i in range(self.horizon):
            encoded.append(self.encoder(inp[:, i].unsqueeze(dim=1)))
        output = self.decoder(torch.stack(encoded, dim=1)).reshape(inp.shape[0], 1, 36, 99)
        if self.act: output = self.act(output)
        return output
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class Decoder(torch.nn.Module):
    def __init__(self, enc_dim, debug = False):
        super(Decoder, self).__init__()
        self.d1 = dcgan_upconv(1, 2, stride=1)
        self.d2 = dcgan_upconv(2, 4, stride=1)
        self.d3 = dcgan_upconv(4, 6, stride=1)
        self.d4 = dcgan_upconv(6, 8, stride=1)
        self.d5 = dcgan_upconv(8, 10, stride=1)
        self.d6 = dcgan_upconv(10, 12, stride=1)
        self.linear = nn.Linear(12*enc_dim, 36*99)
        self.debug = debug

    def forward(self, inp):
        bs, enc_dim = inp.shape
        if self.debug: print(inp.shape)
        inp = inp.reshape(bs, 1, 16, -1)
        d1 = self.d1(inp)
        if self.debug: print(d1.shape)
        d2 = self.d2(d1)
        if self.debug: print(d2.shape)
        d3 = self.d3(d2)
        if self.debug: print(d3.shape)
        d4 = self.d4(d3)
        if self.debug: print(d4.shape)
        d5 = self.d5(d4)
        if self.debug: print(d5.shape)
        d6 = self.d6(d5)
        if self.debug: print(d6.shape)
        out = d6.reshape(bs, 1, 12*d6.shape[2]*d6.shape[3])
        return self.linear(out)

# patch_size - 9, dim - 768, depth - 12, heads - 12, mlp_dim  - 3072, dropout - 0.1, emb_dropout - 0.1 - default setup

# best setup so far patch_size - 9, dim -768, depth - 4, heads - 4, mlp_dim - 1024, dropout = emb_dropout - 0.1

class AutoEncoder(torch.nn.Module):
    def __init__(self, enc_dim, horizon = 1, act = False):
        super(AutoEncoder, self).__init__()
        self.encoder = ViT(
                            image_size = (36, 99),
                            patch_size = 9,
                            num_classes = enc_dim,
                            dim = 768,
                            depth = 4,
                            heads = 4,
                            mlp_dim = 1024,
                            dropout = 0,
                            emb_dropout = 0, 
                            channels = horizon
                            )
    
        self.decoder = Decoder(enc_dim)
        self.horizon = horizon
        self.act = nn.ReLU() if act else None

    def forward(self, inp):
        inp = inp.to(DEVICE, non_blocking = True)
        enc = self.encoder(inp)
        output = self.decoder(enc).reshape(inp.shape[0], 1, 36, 99)
        if self.act: output = self.act(output)
        return output
    
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

    parser.add_argument("--stride", type = int, default = 8)

    parser.add_argument("--horizon", type = int, default = 0)

    parser.add_argument("--act_func", type = int, default = 0)

    # Set defaults 
    args = parser.parse_args()
    arg_dict = vars(args)
    
    # Settings
    print("CONFIG \n", json.dumps(arg_dict, indent=4))
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE: ", DEVICE)

    data_path = "~/mocat-ml/data/TLE_density_all_"
    data = np.load(Path(data_path + f'{args.dataset}.npy').expanduser(), 
                   mmap_mode='c')
    
    data = data[:, :None][:,::args.stride,:].reshape(-1, 1, *(data.shape[2:]))
    data = torch.tensor(data, dtype=torch.float32)

    indices = [i for i in range(data.shape[0])]
    random.shuffle(indices)

    x = int(len(indices)*args.split)
    train_indices, val_indices = indices[:x], indices[x:]
    print(len(train_indices), len(val_indices))

    train_ds = CustomDataset(data[train_indices], horizon = args.horizon)
    val_ds = CustomDataset(data[val_indices], horizon = args.horizon)

    horizon, act = args.horizon if args.horizon != 0 else 1,  True if args.act_func else False
    AE = AutoEncoder(args.enc_dim, horizon = horizon, act = act).to(DEVICE)
    print("AE horizon", AE.horizon)
    
    dls = DataLoaders.from_dsets(train_ds, val_ds, bs = args.bs, num_workers = 1, pin_memory=True, shuffle = True)
    learn = Learner(dls, AE, loss_func = F.mse_loss, cbs = [ShowGraphCallback()])
    print(f'Model size {AE.size()} number of workers: {dls.num_workers}')
    # print(learn.summary())
    
    lr = learn.lr_find().valley
    print(f'lr {round(lr, 5)} \n')
    learn.fit_one_cycle(args.n_epoch)
    # learn.fit(args.n_epoch, lr = 1e-2)
