import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from sample_ddpm import sample_ddpm, visualize_samples_mnist
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import random

# Import custom modules
# from model_unet import UNet, SinusoidalPositionEmbeddings
from model_unet_advanced import UNet, SinusoidalPositionEmbeddings
from diffusion_class import Diffusion

def calculate_inception_score(samples, dataset, device, batch_size=32, splits=10):
    """
    Calculates the Inception Score for generated samples.
    Inspired by: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    and paper: https://papers.nips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf

    Args:
        samples (torch.Tensor): Generated samples with shape (num_samples, channels, height, width).
        device (torch.device): Device to perform calculations on
        dataset (str): One of the two datasets
        batch_size (int): Batch size for processing samples.
        splits (int): Number of splits for calculating the mean and standard deviation of the score.

    Returns:
        float, float: Mean and standard deviation of the Inception Score.
    """

    # Load pretrained InceptionV3 model
    inception_model = models.inception_v3(pretrained=True)  # Keep the final classification layer
    inception_model = inception_model.to(device)

    inception_model.eval()
    samples = samples.to(device)
    num_samples = samples.size(0)

    # Placeholder for storing predictions
    preds = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = samples[i:i + batch_size]
            if dataset == "MNIST":
                batch = batch.repeat(1, 3, 1, 1)  # Repeat channels for Inception
            batch = transforms.functional.resize(batch, size=(299, 299))  # Resize for Inception
            batch = transforms.functional.normalize(batch,
                mean=torch.tensor([0.485, 0.456, 0.406]).to(device),
                std=torch.tensor([0.229, 0.224, 0.225]).to(device),
            )
            pred = inception_model(batch)
            pred = F.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Compute IS
    scores = []
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size: (i + 1) * split_size] # probability of labels conditioned on image
        p_y = np.mean(part, axis=0)
        kl_div = part * (np.log(part) - np.log(p_y[None, :]))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))

    return np.mean(scores), np.std(scores)


# Example Usage
if __name__ == "__main__":
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
    samples = sample_ddpm(unet, diffusion, time_embedding, device, num_samples=1000, dataset='MNIST')
    print("Samples generated successfully!")
    print(np.shape(samples))


    # Calculate Inception Score
    mean_is, std_is = calculate_inception_score(samples, "MNIST", device)
    print(f"Inception Score: {mean_is} ± {std_is}")
