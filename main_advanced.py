'''
This file contains the main script that will be used for training, and subsequently sampling from the DDPM framework.
It imports the necessary functions and classes from the other scripts, including the train and sample functions. 
'''

# Import necessary libraries 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import wandb 

# Import custom modules
from model_unet_advanced import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion
from train_ddpm import train_ddpm_epoch
from sample_ddpm import sample_ddpm
from dataloader import get_dataloader
from custom_lr_scheduler import WarmUpPiecewiseConstantSchedule

## Argument parser
parser = argparse.ArgumentParser(description='Train (and sample from) DDPM framework.')

## Add arguments
parser.add_argument('--T', type=int, default=1000, help='Total timesteps.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (MNIST or CIFAR10).')
parser.add_argument('--save_model', type=bool, default=True, help='Save model after training.')
parser.add_argument('--wandb', default="online", type=str, choices=["online", "disabled"] , help="whether to track with weights and biases or not")
parser.add_argument('--heads', type=int, default=4, help='Number of heads for attention mechanism.')
parser.add_argument('--noise_scheduler', type=str, default='cosine', choices=["linear", "cosine"], help='Noise scheduler type.')
parser.add_argument('--lr_scheduler', type=str, default='none', choices=["none", "warmup_linear"], help='Learning rate scheduler type.')

def main():
    # Parse arguments
    args = parser.parse_args()

    # Unpack arguments
    T = args.T
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    dataset = args.dataset
    save_model = args.save_model
    heads = args.heads
    noise_scheduler = args.noise_scheduler
    lr_scheduler = args.lr_scheduler

    # Scheduler parameters
    warm_up_epochs = 2
    if dataset == 'MNIST':
        lr_decay_epochs = [20, 40, 60] 
    elif dataset == 'CIFAR10':
        lr_decay_epochs = [200, 400, 500] # Tbh not really sure what to put here (qualified guess)


    save_dir = "./saved_models"  # Directory to save the trained model

    # Set number of input channels
    num_input_channels = 1 if dataset == 'MNIST' else 3

    # Set mode for Weights and Biases
    mode_for_wandb = args.wandb
    run_name = f"{dataset}_bs_{batch_size}_Nscheduler_{noise_scheduler}_heads_{heads}"

    # Initialize Weights and Biases
    wandb.init(project='ddpm', entity='dl_ddpm', mode=mode_for_wandb, name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"Using Device: {device}")

    # Dataloader
    train_loader, _ = get_dataloader(dataset, batch_size=batch_size)

    # Initialize components 
    diffusion = Diffusion(T=T, beta_min=10e-5, beta_max=0.02, schedule=noise_scheduler, device=device) 

    time_embedding = SinusoidalPositionEmbeddings(total_time_steps=T, time_emb_dims=128, time_emb_dims_exp=512).to(device)

    model = UNet(input_channels=num_input_channels, 
                 resolutions=[64, 128, 256, 512], 
                 time_emb_dims=512, 
                 dropout=0.1, 
                 use_attention=[False, True, False], 
                 heads=heads).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if lr_scheduler == "warmup_linear":
        scheduler = WarmUpPiecewiseConstantSchedule(optimizer=optimizer, steps_per_epoch=len(train_loader), base_lr=args.lr, 
                                                    lr_decay_ratio=0.2, lr_decay_epochs=lr_decay_epochs, warmup_epochs=warm_up_epochs)
    else:
        scheduler = None



    # for name, param in time_embedding.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_ddpm_epoch(model, diffusion, time_embedding, train_loader, epoch, device, optimizer, scheduler)

        if epoch % 5 == 0:
            # Generate samples 
            samples = sample_ddpm(model, diffusion, time_embedding, device, num_samples=2, dataset=dataset) 
            wandb.log({"Generated Samples": [wandb.Image(sample, caption=f"Epoch {epoch}") for sample in samples]})

    if save_model:
        final_save_path = f"{save_dir}/ddpm_{dataset}_{noise_scheduler}_heads_{heads}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            "embedding_state_dict": time_embedding.state_dict(),
        }, final_save_path)

        print(f"Model and embedding saved at: {final_save_path}")

    # Finish Weights and Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
    