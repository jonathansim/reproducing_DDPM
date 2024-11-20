import matplotlib.pyplot as plt
import numpy as np
import torch
 
from diffusion_class import Diffusion
from dataloader import get_dataloader

def visualize_diffusion(diffusion: object, x0: torch.Tensor, timesteps: list):
    """
    Visualize the diffusion process.
    Args:
        diffusion: Diffusion object.
        x0: Original data (batch)
        timesteps: List of timesteps to visualize.
    """

    num_images = min(5, x0.size(0))  # Visualize up to 5 images
    fig, axs = plt.subplots(num_images, len(timesteps), figsize=(13, 1 * num_images))

    for i, t in enumerate(timesteps):
        # Create a tensor of timesteps for the batch
        t_tensor = torch.full((x0.size(0),), t, dtype=torch.long, device=x0.device)

        # Get the noisy version of x0 at timestep t
        xt, _ = diffusion.forward_diffusion(x0, t_tensor)

        for j in range(num_images):
            # Visualize each image
            img = xt[j].detach().cpu().numpy().squeeze()
            axs[j, i].imshow(img, cmap="gray")
            if j == 0:
                axs[j, i].set_title(f"t={t}")
            axs[j, i].axis("off")
    
    plt.tight_layout()
    plt.show()




# Test the visualization
if __name__ == "__main__":
    # Data loader
    train_dataloader, _ = get_dataloader(dataset="MNIST", batch_size=3)
    x0, _ = next(iter(train_dataloader)) # Get a batch of images

    # Ensure x0 is in the correct shape and range
    # x0 = x0.to(torch.float32)  # Convert to float
    # if x0.max() > 1:  # Normalize if necessary
    #     x0 /= 255.0
    # x0 = x0.unsqueeze(1)  # Add channel dimension if missing

    # print(x0.min(), x0.max(), x0.shape)

    # Initialize diffusion process
    diffusion = Diffusion(T=1000, beta_min=0.0001, beta_max=0.02, schedule='linear')

    # Visualize diffusion at selected timesteps
    timesteps = [0, 25, 50, 100, 150, 200, 300, 400, 650, 800, 999]  # Selected timesteps
    visualize_diffusion(diffusion, x0, timesteps)