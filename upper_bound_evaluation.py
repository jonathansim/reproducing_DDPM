import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from fid import calculate_fid, compute_activations, get_images_in_batches, get_inception_model, get_real_image_activations, calculate_fid_generated_samples
from inception_score import calculate_inception_score_from_dataloader
import wandb
from dataloader import get_dataloader_evaluation
import random
import math

torch.cuda.empty_cache()

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

data = "CIFAR10"
print(data)
inception_model = get_inception_model()
real_activations = get_real_image_activations(inception_classifier=inception_model, data=data, num_images=50000)
fid = calculate_fid(real_activations[:25000], real_activations[25000:])
print("Fid: ", fid)

torch.cuda.empty_cache()

dataloader = get_dataloader_evaluation(data, batch_size = 100)
mean_is, std_is = calculate_inception_score_from_dataloader(dataloader=dataloader, dataset=data, device=device)
print(f"Inception Score: {mean_is} ± {std_is}")
