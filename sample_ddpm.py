'''
This script will contain algorithm 2 from the paper, which is the sampling procedure for the DDPM framework.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Import custom modules
# from model_unet import UNet, SinusoidalPositionEmbeddings
from model_unet_advanced import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion



def sample_ddpm(model: object, diffusion: object, time_embedding: object, device: str, num_samples: int = 16, dataset: str = 'MNIST'):
    """
    This function implements Algorithm 2 from the paper, specifically it samples from the U-Net model.
    Args:
        model: U-Net model.
        diffusion: Diffusion object.
        time_embedding: Time embedding object.
        device: Device to use (e.g., 'cuda' or 'cpu').
        num_samples: Number of samples to generate.
        dataset: Dataset to use ('MNIST' or 'CIFAR10').
    Returns:
        samples: Generated samples.
    """

    print(f"Sampling {num_samples} samples...")

    model.eval()
    with torch.no_grad():
        # 1. Initialize samples from standard gaussian distribution
        if dataset == 'MNIST':
            x = torch.randn((num_samples, 1, 28, 28), device=device)
        elif dataset == 'CIFAR10':
            x = torch.randn((num_samples, 3, 32, 32), device=device)

        # 2. iterative reverse diffusion 
        for t in reversed(range(1, diffusion.T + 1)):
            # prepare timestep tensor
            t_tensor = torch.full((num_samples,), t - 1, device=device, dtype=torch.long)

            # Perform reverse diffusion step
            x = diffusion.reverse_diffusion(x, t_tensor, model, time_embedding)
        
        # 3. Return samples
        return x

def visualize_samples_mnist(samples): 
    """
    Visualize the generated samples.
    Args:
        samples: Generated samples.
    """
    num_samples = samples.size(0)
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 20))

    for i in range(num_samples):
        sample = samples[i].cpu().numpy().squeeze()
        axs[i].imshow(sample, cmap='gray')
        axs[i].axis('off')

    plt.show()


def visualize_samples_cifar10(samples, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_cols=math.ceil(math.sqrt(25))): 
    """
    Visualize the generated samples in multiple rows.
    Args:
        samples: Generated samples of shape (num_samples, 3, 32, 32).
        mean: Tuple of mean values used for normalization (default for CIFAR is (0.5, 0.5, 0.5)).
        std: Tuple of std values used for normalization (default for CIFAR is (0.5, 0.5, 0.5)).
        num_cols: Number of images per row.
    """
    # Denormalize the images
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(samples.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(samples.device)
    samples = samples * std + mean  # Denormalize

    # Clip values to [0, 1] range for display
    samples = samples.clamp(0, 1)

    num_samples = samples.size(0)
    num_rows = math.ceil(num_samples / num_cols)  # Calculate the number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Flatten axs array for easier iteration, even if it's a single row
    axs = axs.flatten()

    for i in range(num_samples):
        sample = samples[i].permute(1, 2, 0).cpu().numpy()  # Convert (3, 32, 32) -> (32, 32, 3)
        axs[i].imshow(sample)
        axs[i].axis('off')

    # Turn off axes for unused subplots
    for j in range(num_samples, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_samples_mnist(samples, mean=(0.5), std=(0.5), num_cols=math.ceil(math.sqrt(25))): 
    """
    Visualize the generated samples in multiple rows.
    Args:
        samples: Generated samples of shape (num_samples, 3, 32, 32).
        mean: Tuple of mean values used for normalization (default for CIFAR is (0.5, 0.5, 0.5)).
        std: Tuple of std values used for normalization (default for CIFAR is (0.5, 0.5, 0.5)).
        num_cols: Number of images per row.
    """
    # Denormalize the images
    mean = torch.tensor(mean).view(1, 1, 1, 1).to(samples.device)
    std = torch.tensor(std).view(1, 1, 1, 1).to(samples.device)
    samples = samples * std + mean  # Denormalize

    # Clip values to [0, 1] range for display
    samples = samples.clamp(0, 1)

    num_samples = samples.size(0)
    num_rows = math.ceil(num_samples / num_cols)  # Calculate the number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Flatten axs array for easier iteration, even if it's a single row
    axs = axs.flatten()

    for i in range(num_samples):
        sample = samples[i].permute(1, 2, 0).cpu().numpy()  # Convert (3, 32, 32) -> (32, 32, 3)
        axs[i].imshow(sample, cmap='gray')
        axs[i].axis('off')

    # Turn off axes for unused subplots
    for j in range(num_samples, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the sampling procedure#

    # Set device 
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"Using Device: {device}")

    # Step 1: Initialize the Sinusoidal Embeddings
    time_embedding = SinusoidalPositionEmbeddings(total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512).to(device)
    
    # Step 2: Initialize the U-Net and Diffusion
    unet = UNet(input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512, dropout=0.1, use_attention=[False, True, False], heads=4).to(device)
    diffusion = Diffusion(T=1000, beta_min=10e-5, beta_max=0.02, schedule='cosine', device=device)
    
    # Step 3: Load the trained model
    #model_path = "ddpm_CIFAR10_linear_heads_1_LRs_none_seed7.pth"
    model_path = "ddpm_MNIST_cosine_final.pth"
    saved = torch.load(model_path, map_location=device)

    unet.load_state_dict(saved["model_state_dict"])
    time_embedding.load_state_dict(saved["embedding_state_dict"])

    # Number of params (TEST)
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Step 4: Generate samples
    samples = sample_ddpm(unet, diffusion, time_embedding, device, num_samples=28, dataset='MNIST')
    print("Samples generated successfully!")
    #visualize_samples_cifar10(samples, num_cols=8)
    visualize_samples_mnist(samples, num_cols=7)
    