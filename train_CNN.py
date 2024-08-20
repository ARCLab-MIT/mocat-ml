import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import torch
import torchvision
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from CNN import *
import random, os, h5py, json

data_dir = '/home/gridsan/ssarangerel/data/mocatml/TLE_density_all.mat'
batch_size = 8192
split = 0.95
lr=0.001
d = 128
epochs = 30
num_sim = 100

with open("data_config.json", "r") as outfile:
    data = json.load(outfile, strict=False)
    
extent = list(data['extent'])
data_numpy = np.load('data.npy')
                     
indices = [i for i in range(num_sim)]
random.shuffle(indices)
train_indices = indices[:int(split * num_sim)]
val_indices = indices[int(split * num_sim):]

train_data = CustomDataset(data_numpy, train_indices)
val_data = CustomDataset(data_numpy, val_indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

loss_fn = torch.nn.MSELoss()

encoder = Encoder(d)
decoder = Decoder(d)

params_to_optimize = [{'params': encoder.parameters()}, {'params': decoder.parameters()}]
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

encoder.to(device)
decoder.to(device)

print("Training")
train_losses, val_losses = [], []

dir_name = f'new_epochs_{epochs}_d_{d}'
if not os.path.isdir(dir_name):
    os.mkdir(os.path.join(os.getcwd(), dir_name))

for epoch in range(epochs):
    train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)
    train_losses += train_loss

    n = len(train_loss)

    val_loss = test(encoder, decoder, device, valid_loader, loss_fn).item()
    val_losses.append(val_loss)
    print(f'Epoch {epoch+1}')
    print('Validation loss', val_loss)

    idx = random.randint(0, len(val_data))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    img = val_data[idx].unsqueeze(0).to(device)
    encoder.eval()
    decoder.eval()
    recon = decoder(encoder(img)).cpu().detach().numpy()

    axs[0].imshow(val_data[idx], extent = extent, aspect = 'auto', label = 'original')
    axs[1].imshow(recon[0], extent = extent, aspect = 'auto', label = 'reconstruction')
    axs[0].set_title('original')
    axs[1].set_title('reconstruction')
    fig.suptitle(f'Epoch {epoch+1} Validation loss {val_loss}')
    plt.savefig(f'{dir_name}/{epoch+1}.jpg')
    plt.close(fig)

plt.figure() 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(range(len(train_losses)), train_losses)
axs[1].plot(range(len(val_losses)), val_losses)
axs[0].set_title('Training loss')
axs[1].set_title('Validation loss')
# for i in range(epochs):
#     plt.axvline(x = (i+1) * n)
plt.savefig(f'train_val_loss/epoch_{epochs}_bsize_{batch_size}_d_{d}.jpg')
