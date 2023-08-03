import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py, random
from torch.utils.data import Dataset


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
        )

        # self.flatten = nn.Flatten(start_dim=-2)
        self.flatten = nn.Sequential(
                nn.Flatten(start_dim=-2), 
                nn.Flatten(start_dim=-2)
        )

        self.encoder_lin = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Linear(512, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.Linear(512, 1536),
        )

        self.unflatten = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(32, 48)), 
            nn.Unflatten(dim=-1, unflattened_size=(4, 12))
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=(1, 0))
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = Encoder(d)
        self.decoder = Decoder(d)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train_epoch(self, device, dataloader, loss_fn, optimizer):
        self.train()
        train_loss = []
        for _, image_batch in dataloader:
            image_batch = image_batch.to(device)
            recon_batch = self(image_batch)
            loss = loss_fn(recon_batch, image_batch)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
        return train_loss
    
    def test(self, device, dataloader, loss_fn, optimizer):
        self.eval()
        val_loss = []
        with torch.no_grad(): # No need to track the gradients
            for _, image_batch in dataloader:
                image_batch = image_batch.to(device)
                recon_batch = self(image_batch.to(device))
                val_loss.append(loss_fn(recon_batch, image_batch).item())
        return val_loss
    
    
# tsai library dataset - tsdataset or tsdataloader -  for sequences
class CustomDataset(Dataset):
    def __init__(self, data_numpy, indices, device):
        self.data = data_numpy[indices]
        self.data = torch.tensor(self.data, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
        self.data = self.data.unsqueeze(0).transpose(0, 1)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader:
        # print(image_batch.shape)
        # image_batch = image_batch.to(device).unsqueeze(0).transpose(0, 1)
        # print(image_batch.shape)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        train_loss.append(loss.item())
    return train_loss

### Testing function
def test(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device).unsqueeze(0).transpose(0, 1)

            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

