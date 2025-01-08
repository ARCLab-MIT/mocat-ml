import torch, os
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

import fastai
from fastai.data.transforms import RandomSplitter

import denoising_diffusion_pytorch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.attend import Attend
from ema_pytorch import EMA

import copy, logging, argparse, wandb, datetime, gc, json, random, math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from diffusion_utils import *


def train(config):

    train_dl, val_dl, data, data_sw, splits, val_ds, max_val, min_val = get_dataloader_diffusion(config) 
    vmin, vmax = math.exp(min_val), math.exp(max_val)
    device = config.device

    dim_mults = (1, 2, 4, 8, 16) if config.unet_size == 2 else (1, 2, 4, 8)
    if config.unet_size == 0:
        dim_mults = (1, 2, 4)

    model = Unet(
            dim = config.unet_dim,
            dim_mults = dim_mults,
            flash_attn = False, # does not work when flash_attn=True "No kernel found error"
            channels = config.horizon
    ).to(device)


    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = config.timesteps, 
        objective = 'pred_x0'
    ).to(device)

    print("MODEL SIZE: ", get_n_params(model))

    # wandb.init(project="MOCAT_diffusion", entity="sumiya")

    # wandb.config = {"# Epochs" : config.n_epoch,
    #                 "Batch size" : config.bs,
    #                 "Image size" : config.image_size,
    #                 "Device" : config.device,
    #                 "Lr" : config.lr}

    # wandb.watch(model, log=None)

    date, c = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now()), config
    save_to = f"diffusion_results/{date}_l{c.lookback}_s{c.stride}_ds{c.ds}_loss_{c.loss}_bs{c.bs}_avg_{c.average}_log_{c.log}_norm_{c.normalize}"
    save_to += f"_unet{c.unet_size}_dim{c.unet_dim}"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # Define optimizer
    optimizer = Adam(diffusion.parameters(), lr=config.lr)
    model.train()

    # Training loop
    train_losses, val_losses = [], []
    for epoch in range(config.n_epoch):
        diffusion.train()

        train_loss = 0
        for batch in train_dl:
            lkb, hor = batch

            lkb = lkb.float().to(device)  
            hor = hor.float().to(device)

            loss = diffusion(lkb, hor, config.loss)
            train_loss += loss.item()

            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Print progress
        train_loss /= len(train_dl)
        train_loss = unnormalize_neg_one_to_one(train_loss, max_val, min_val)
        train_loss = math.exp(train_loss)-1 if config.normalize else train_loss
        train_losses.append(train_loss)

        final_loss = unnormalize_neg_one_to_one(loss.item(), max_val, min_val)
        final_loss = math.exp(final_loss)-1 if config.normalize else final_loss

        print(f"Epoch [{epoch+1}/{config.n_epoch}], Training Loss: avg {train_loss:.4f} final: {final_loss:.4f}")

        # Evaluate on validation set
        with torch.no_grad():
            diffusion.eval()
            val_loss = 0.0
            for val_batch in val_dl:
                val_lkb, val_hor = val_batch

                val_lkb = val_lkb.float().to(device)  # Assuming you're using GPU
                val_hor = val_hor.float().to(device)
                val_loss += diffusion(val_lkb, val_hor, config.loss).item()

            val_loss /= len(val_dl)
            val_loss = unnormalize_neg_one_to_one(val_loss, max_val, min_val)
            val_loss = math.exp(val_loss)-1 if config.normalize else val_loss
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

            # wandb.log({
            #     "Training Loss": train_loss / len(train_dl),
            #     "Validation Loss": val_loss / len(val_dl),
            #     'Sampled images': wandb.Image(plt)
            # })
        
            # Logging and saving
            if (epoch+1)%1 == 0 and epoch >= 0:
                print("Doing one step ahead predictions ... ")
                
                save_path = f"{save_to}/1 step ahead predictions"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                random_index = torch.randint(0, len(val_ds), (1,)).item()
                sample_horizon, sample_lookback =  val_ds[random_index]
                sample_horizon, sample_lookback = torch.tensor(sample_horizon).float().to("cuda"), torch.tensor(sample_lookback).float().to("cuda")
                sample_horizon, sample_lookback = torch.unsqueeze(sample_horizon, 0), torch.unsqueeze(sample_lookback, 0)

                pure_noise = torch.rand_like(sample_horizon)*2 - 1
                x = torch.cat((sample_lookback, pure_noise), dim = 3)
                pred = diffusion.ddim_sample_w_start(x_start=x, shape=x.shape)
                target = torch.cat((sample_lookback, sample_horizon), dim = 3)
                
                x, lkb_hor, lkb_pred = x[0][0], target[0][0], pred[0][0]
                x, lkb_hor, lkb_pred = unnormalize_neg_one_to_one(x, max_val, min_val), unnormalize_neg_one_to_one(lkb_hor, max_val, min_val), unnormalize_neg_one_to_one(lkb_pred, max_val, min_val)
                if config.log:
                    x, lkb_hor, lkb_pred = torch.exp(x)-1, torch.exp(lkb_hor)-1, torch.exp(lkb_pred)-1
                    relu = nn.ReLU()
                    x, lkb_hor, lkb_pred = relu(x), relu(lkb_hor), relu(lkb_pred)
                
                w = x.shape[1]//2
                inp, pred, target = x[:, :w], lkb_pred[:, w:], lkb_hor[:, w:]
                plot(inp, pred, target, f"{save_path}/after {epoch} epochs", vmin, vmax)


            if epoch+1 == config.n_epoch:
                print("Doing long term prediction ...")
                
                idx = random.choice(splits[1])
                sample_data = torch.from_numpy(data[idx]).float()
                num_iterations = 2436//(config.lookback  + config.gap) - 1
                inp_pred = sample_data[:config.lookback].to("cuda")
                predictions = torch.zeros((num_iterations+1, inp_pred.shape[1], inp_pred.shape[2]*2))
                x = relu(torch.exp(unnormalize_neg_one_to_one(inp_pred[0], max_val, min_val))-1)   
                predictions[0] = torch.cat((x, x), dim = 1)
                relu = nn.ReLU()

                save_path = f"{save_to}/long_term_prediction_after_{epoch+1}_epochs"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for iter in range(num_iterations):
                    inp, target = inp_pred.to("cuda"), sample_data[(iter+1)*config.lookback: (iter+2)*config.lookback].to("cuda")
                    inp, target = torch.unsqueeze(inp, 0), torch.unsqueeze(target, 0)

                    pure_noise = torch.rand_like(target)*2-1
                    inp_noise = torch.cat((inp, pure_noise), dim = 3)
                    inp_target = torch.cat((inp, target), dim = 3)
                    inp_pred = diffusion.ddim_sample_w_start(x_start=inp_noise, shape=inp_noise.shape)
                    nxt = inp_pred

                    if iter == 0 or iter%20 == 0:
                        inp_noise, inp_target, inp_pred = inp_noise[0][0], inp_target[0][0], inp_pred[0][0]
                        inp_noise, inp_target, inp_pred = unnormalize_neg_one_to_one(inp_noise, max_val, min_val), unnormalize_neg_one_to_one(inp_target, max_val, min_val), unnormalize_neg_one_to_one(inp_pred, max_val, min_val)
                        
                        if config.log:
                            inp_noise, inp_target, inp_pred = torch.exp(inp_noise)-1, torch.exp(inp_target)-1, torch.exp(inp_pred)-1
                            inp_noise, inp_target, inp_pred = relu(inp_noise), relu(inp_target), relu(inp_pred)

                        w = inp_noise.shape[1]//2
                        inp_noise, inp_pred, inp_target = inp_noise[:, :w],  inp_pred[:, w:], inp_target[:, w:]
                        plot(inp_noise, inp_pred, inp_target, save_path + f"/forecast_iteration_{iter}", vmin, vmax)
                        

                    inp_pred = nxt[0, :, :, nxt.shape[-1]//2:]
                    x, y = relu(torch.exp(unnormalize_neg_one_to_one(target[0][0], max_val, min_val))-1), relu(torch.exp(unnormalize_neg_one_to_one(inp_pred[0], max_val, min_val))-1)
                    predictions[iter+1] = torch.cat((x, y), dim = 1)
 
                make_and_save_gif(predictions, f"{save_path}/predictions.gif", vmin, vmax)
                w = predictions.shape[2]//2
                trgs, preds = predictions[:, :, w:], predictions[:, :, :w]
                print(trgs.shape, preds.shape)
                losses = torch.mean(torch.pow(trgs - preds, 2), dim = [1,2]) if config.loss == "mse" else torch.mean(torch.abs(trgs - preds), dim = [1,2])   # mse loss
                print(losses.shape)
                
                plt.figure(figsize=(8, 5))  
                plt.plot(losses, label='Loss', color='blue')
                plt.xlabel('Forecast Iteration')
                plt.ylabel(f'Loss ({config.loss})')
                plt.title('Loss Evolution During Long Term Prediction')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.savefig(save_path + "/losses.png")
                plt.close()

                # torch.save(diffusion.state_dict(), f'diffusion_model_epoch_{epoch+1}.pth')

    #plot train and val losses
    plot_losses(train_losses, val_losses, f"{save_to}/losses.png")

    #save losses
    with open(f"{save_to}/train_losses.txt", "w") as output:
        output.write(str(train_losses))

    with open(f"{save_to}/val_losses.txt", "w") as output:
        output.write(str(val_losses))

    # Save final model
    torch.save(diffusion.state_dict(), 'diffusion_model_final.pth')
    print("Training Done")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type = str, default = "x8x8", help = "dataset to train on")     
    parser.add_argument("--image_size", type = int, default = 32)
    parser.add_argument("--horizon", type = int, default = 1) 
    parser.add_argument("--lookback", type = int, default = 1) 
    parser.add_argument("--gap", type = int, default = 0) 
    parser.add_argument("--stride", type = int, default = 8) 
    parser.add_argument("--bs", type = int, default = 32)
    parser.add_argument("--n_epoch", type = int, default = 100)
    parser.add_argument("--loss", type = str, default='mse')
    parser.add_argument("--key", type = str, default = 'comb_Am_rp')
    parser.add_argument("--sample", type = int, default = 1) # whether or not to sample to 32x32
    parser.add_argument("--device", type = str, default = "cuda"),
    parser.add_argument("--lr", type = float, default = 3e-4)
    parser.add_argument("--run_name", type = str, default = "mocat-ml_diffusion")
    parser.add_argument("--log", type=int, default=1) #whether or not to learn log(N)
    parser.add_argument("--train_on_diff", type=int, default=0) # whether or not to train on the difference $TODO
    parser.add_argument("--average", type=int, default=0) #training on averaged data
    parser.add_argument("--large_model", type = int, default=0) #TODO
    parser.add_argument("--diff", type = int, default=0)
    parser.add_argument("--normalize", type = int, default = 1) #1 - True 0 - False
    parser.add_argument("--unet_size", type = int, default = 0) #2 - Huge 1 - Big 0 - Small
    parser.add_argument("--unet_dim", type = int, default = 16)
    parser.add_argument("--timesteps", type = int, default = 1000)
    
    args = parser.parse_args()
    train(args)

