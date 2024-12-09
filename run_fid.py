from sample_ddpm import sample_ddpm, visualize_samples_mnist
from fid import full_fid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random

# Import custom modules
# from model_unet import UNet, SinusoidalPositionEmbeddings
from model_unet_advanced import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion

#torch.cuda.empty_cache()

def set_training_seed(seed):
    # Function to set the different seeds 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_training_seed(7)


device = torch.device("cuda" if torch.cuda.is_available() else 
                    "mps" if torch.backends.mps.is_available() else 
                    "cpu")
print(f"Using Device: {device}")

# Step 1: Initialize the Sinusoidal Embeddings
time_embedding = SinusoidalPositionEmbeddings(total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512).to(device)

# Step 2: Initialize the U-Net and Diffusion
unet = UNet(input_channels=1, resolutions=[64, 128, 256, 512], time_emb_dims=512, dropout=0.1, use_attention=[False, True, False], heads=8).to(device)
diffusion = Diffusion(T=1000, beta_min=10e-5, beta_max=0.02, schedule='cosine', device=device)

# Step 3: Load the trained model
model_path = "saved_models/ddpm_MNIST_linear_heads_8_LRs_none_seed7.pth"
print("Model_path: ", model_path)
saved = torch.load(model_path, map_location=device)

unet.load_state_dict(saved["model_state_dict"])
time_embedding.load_state_dict(saved["embedding_state_dict"])


# Step 4: Generate samples
samples = sample_ddpm(unet, diffusion, time_embedding, device, num_samples=10000, dataset='MNIST')
print("Samples generated successfully!")
print(np.shape(samples))

fid = full_fid(samples, data = "MNIST", num_images = 10000)
#fid = full_fid_mnist_base()
print("Fid: ", fid)
