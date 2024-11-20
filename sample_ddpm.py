'''
This script will contain algorithm 2 from the paper, which is the sampling procedure for the DDPM framework.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def sample_ddpm(model: object, diffusion: object, time_embedding: object, device: str, num_samples: int = 16, dataset: str = 'mnist'):
    """
    This function implements Algorithm 2 from the paper, specifically it samples from the U-Net model.
    Args:
        model: U-Net model.
        diffusion: Diffusion object.
        time_embedding: Time embedding object.
        device: Device to use (e.g., 'cuda' or 'cpu').
        num_samples: Number of samples to generate.
        dataset: Dataset to use ('mnist' or 'cifar10').
    Returns:
        samples: Generated samples.
    """

    print(f"Sampling {num_samples} samples...")

    model.eval()
    with torch.no_grad():
        # 1. Initialize samples from standard gaussian distribution
        if dataset == 'mnist':
            x = torch.randn((num_samples, 1, 28, 28), device=device)
        elif dataset == 'cifar10':
            x = torch.randn((num_samples, 3, 32, 32), device=device)

        # 2. iterative reverse diffusion 
        for t in reversed(range(1, diffusion.T + 1)):
            # prepare timestep tensor
            t_tensor = torch.full((num_samples,), t - 1, device=device, dtype=torch.long)

            # Perform reverse diffusion step
            x = diffusion.reverse_diffusion(x, t_tensor, model, time_embedding)
        
        # 3. Return samples
        return x
    