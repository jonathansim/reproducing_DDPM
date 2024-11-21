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
from model_unet import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion
from train_ddpm import train_ddpm_epoch
from sample_ddpm import sample_ddpm
from dataloader import get_dataloader

# Argument parser
# TODO Implement if script is to be run on HPC 

def main():
    # Initialize Weights and Biases
    wandb.init(project='ddpm', entity='dl_ddpm')

    T = 1000  # Total timesteps
    batch_size = 64  # Batch size
    num_epochs = 5  # Number of epochs
    lr = 2e-4  # Learning rate
    dataset = "MNIST"  # Dataset to use ('mnist' or 'cifar10')
    save_model = True  # Save model after training

    save_dir = "./saved_models"  # Directory to save the trained model

    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"Using Device: {device}")

    # Dataloader
    train_loader, _ = get_dataloader(dataset, batch_size=batch_size)

    # Initialize components 
    diffusion = Diffusion(T=T, beta_min=10e-5, beta_max=0.02, schedule='linear', device=device) 
    time_embedding = SinusoidalPositionEmbeddings(total_time_steps=T, time_emb_dims=128, time_emb_dims_exp=512).to(device)
    model = UNet(input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for name, param in time_embedding.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_ddpm_epoch(model, diffusion, time_embedding, train_loader, epoch, device, optimizer)

        if epoch % 5 == 0:
            # Generate samples 
            samples = sample_ddpm(model, diffusion, time_embedding, device, num_samples=2, dataset=dataset) 
            wandb.log({"Generated Samples": [wandb.Image(sample, caption=f"Epoch {epoch}") for sample in samples]})

    if save_model:
        final_save_path = f"{save_dir}/ddpm_{dataset}_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            "embedding_state_dict": time_embedding.state_dict(),
        }, final_save_path)

        print(f"Model and embedding saved at: {final_save_path}")

    # Finish Weights and Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
    print("Training completed successfully!")