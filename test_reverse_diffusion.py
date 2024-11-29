'''
This script will be used to test the different functions defined in the other scripts. 

'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from diffusion_class import Diffusion
from model_unet import SinusoidalPositionEmbeddings, UNet


def test_reverse_diffusion(make_plots=True):
    # Parameters
    T = 1000  # Total timesteps
    image_size = (1, 28, 28)  # Image dimensions (e.g., for MNIST)
    num_samples = 3  # Number of samples to test

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Initialize diffusion and embedding
    diffusion = Diffusion(T=T, beta_min=1e-5, beta_max=0.02, schedule='cosine', device=device)
    # diffusion.to(device)

    time_embedding = SinusoidalPositionEmbeddings(total_time_steps=T, time_emb_dims=128, time_emb_dims_exp=512).to(device)

    # Initialize dummy model (an untrained UNet for testing)
    model = UNet(input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512).to(device)
    model.eval()  

    # Generate noisy input x_t
    xt = torch.randn((num_samples, *image_size), device=device)

    # Test reverse diffusion for a random timestep
    ## Random timesteps for the batch
    t = torch.randint(1, T, (num_samples,), device=device, dtype=torch.long)  # Random timesteps for the batch
    
    ## Edge case: Test for t=1
    # t = torch.full((num_samples,), device=device, dtype=torch.long)

    print(f"Testing reverse diffusion for timesteps: {t}")

    with torch.no_grad():
        xt_minus_one = diffusion.reverse_diffusion(xt, t, model, time_embedding)

    print("Reverse Diffusion Successful!")

    # Visualize xt and xt_minus_one
    if make_plots:
        fig, axs = plt.subplots(2, num_samples, figsize=(15, 5))
        for i in range(num_samples):
            axs[0, i].imshow(xt[i].cpu().squeeze(), cmap="gray")
            axs[0, i].set_title(f"x_t (t={t[i].item()})")
            axs[0, i].axis("off")
            
            axs[1, i].imshow(xt_minus_one[i].cpu().squeeze(), cmap="gray")
            axs[1, i].set_title(f"x_(t-1)")
            axs[1, i].axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test the reverse diffusion
    test_reverse_diffusion(make_plots=True)
    print("Reverse diffusion test passed successfully!")
