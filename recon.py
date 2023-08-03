import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import torch
import torchvision
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from CNN import *


data_dir = '/home/gridsan/ssarangerel/data/mocatml/TLE_density_all.mat'

batch_size = 8192
split = 0.95
lr=0.001
d = 256
epochs = 10

train_dataset = CustomDataset(data_dir)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

loss_fn = torch.nn.MSELoss()

encoder = Encoder(d)
decoder = Decoder(d)
params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)

img = train_dataset[0].unsqueeze(0).to(device)
recon = decoder(encoder(img)).cpu().detach().numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(train_dataset[0], extent = train_dataset.extent, aspect = 'auto', label = 'original')
axs[1].imshow(recon[0], extent = train_dataset.extent, aspect = 'auto', label = 'recon')
axs[0].set_title('original')
axs[1].set_title('reconstruction')

fig.suptitle("akjbcka")
plt.savefig('anclnac.jpg')
