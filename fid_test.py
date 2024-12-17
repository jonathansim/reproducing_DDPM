import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
from fid_copy import full_fid, get_inception_model, get_real_image_activations, calculate_fid_generated_samples
torch.cuda.empty_cache()
import random

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 25
latent_size = 100


data = "MNIST"

num_gpu = 1 if torch.cuda.is_available() else 0

# load the models





# load weights

if data == "MNIST":
    from dcgan import Discriminator, Generator
    D = Discriminator(ngpu=1).eval()
    G = Generator(ngpu=1).eval()

    D.load_state_dict(torch.load('netD_epoch_99.pth'))
    G.load_state_dict(torch.load('netG_epoch_99.pth'))
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()

elif data == "CIFAR10":

    from dcgan_cifar import Discriminator, Generator
    D = Discriminator(ngpu=1).eval()
    G = Generator(ngpu=1).eval()

    D.load_state_dict(torch.load('netD_epoch_199.pth'))
    G.load_state_dict(torch.load('netG_epoch_199.pth'))
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()





def generate_fake_images(generator, num_samples=2000, latent_dim=100, device='cuda'):
    generator.eval()
    z = torch.randn(num_samples, latent_dim, 1, 1, device=device)  # Random noise
    with torch.no_grad():
        fake_images = generator(z)  # Shape: [num_samples, 1, 28, 28]
    return fake_images



gen_img = generate_fake_images(G, num_samples=10000, latent_dim=100, device='cuda')

fid = full_fid(gen_img, data=data,num_images=10000)
print(fid)