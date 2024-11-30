'''
This script will train the DDPM framework, using a neural network model (U-Net) to predict the noise in the diffusion process.
This largely corresponds to algorithm 1 in the paper. 
'''
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import wandb


def train_ddpm_epoch(model: object, diffusion: object, time_embedding: object, train_loader: object, epoch: int, device: str, optimizer: object, lr_scheduler: object):
    """
    This function implements Algorithm 1 from the paper, specifically it trains the U-Net model for one epoch.
    Args:
        model: U-Net model.
        diffusion: Diffusion object.
        time_embedding: Time embedding object.
        train_loader: DataLoader for training data.
        epoch: Current epoch.
        device: Device to use (e.g., 'cuda' or 'cpu').
        optimizer: Optimizer for training.
    """

    print(f"Training epoch {epoch}...")

    model.train()

    for i, (x0, _) in enumerate(train_loader):
        x0 = x0.to(device)

        # Sample random timestep
        t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)

        # Get sinusoidal time embedding
        time_emb = time_embedding(t)

        # Forward diffusion
        xt, epsilon = diffusion.forward_diffusion(x0, t)

        # Predict noise
        epsilon_pred = model(xt, time_emb)

        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Use learning rate scheduler
        if lr_scheduler:
            lr_scheduler.step()

        # Log loss
        wandb.log({"loss": loss.item(), "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

        if epoch > 1: 
            wandb.log({"loss after 1st epoch": loss.item()})

        if i % 100 == 0:
            print(f"Epoch {epoch}, batch {i}: loss={loss.item()}")


