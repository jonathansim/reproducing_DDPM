from fid import full_fid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
from inception_score import calculate_inception_score
from dcgan_cifar import Discriminator, Generator

# Import custom modules
# from model_unet import UNet, SinusoidalPositionEmbeddings
from model_unet_advanced import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion

torch.cuda.empty_cache()

def set_training_seed(seed):
    # Function to set the different seeds 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_training_seed(24) # REMEMBER !!!


device = torch.device("cuda" if torch.cuda.is_available() else 
                    "mps" if torch.backends.mps.is_available() else 
                    "cpu")
print(f"Using Device: {device}")

num_gpu = 1 if torch.cuda.is_available() else 0

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

D.load_state_dict(torch.load('netD_epoch_199.pth'))
G.load_state_dict(torch.load('netG_epoch_199.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()


def generate_fake_images(generator, num_samples=10000, latent_dim=100, device=device):
    generator.eval()
    z = torch.randn(num_samples, latent_dim, 1, 1, device=device)  # Random noise
    with torch.no_grad():
        fake_images = generator(z)  # Shape: [num_samples, 1, 28, 28]
    return fake_images



gen_img = generate_fake_images(G, num_samples=10000, latent_dim=100, device=device)
print(np.shape(gen_img))
print("MODEL: DC GAN")
fid = full_fid(gen_img, data="CIFAR10" ,num_images=10000)
print(fid)

mean_is, std_is = calculate_inception_score(samples=gen_img, dataset='CIFAR10', device=device, batch_size=32, splits=10)
print("IS: ", mean_is)
