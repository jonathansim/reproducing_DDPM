from sample_ddpm import sample_ddpm,sample_ddpm_with_intermediates, visualize_samples_mnist
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
from dataloader import get_dataloader

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

def compute_nll_vlb(model, diffusion, data_loader, time_embedding, device):
    """
    Compute the Negative Log-Likelihood (NLL) using the variational lower bound (VLB).
    Args:
        model: Trained DDPM model.
        diffusion: Diffusion process object with precomputed schedules.
        data_loader: DataLoader for evaluation dataset.
        time_embedding: Time embedding module.
        device: Device for computation (e.g., 'cuda').
    Returns:
        Average NLL over the dataset.
    """
    total_nll = 0
    num_samples = 0
    def gaussian_kl_divergence(mean1, logvar1, mean2, logvar2):
        """
        Compute KL divergence between two Gaussian distributions:
        N(mean1, exp(logvar1)) and N(mean2, exp(logvar2)).
        """
        return 0.5 * (
            -1.0
            + logvar2 - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
    def approx_standard_normal_cdf(x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    def discretized_gaussian_log_likelihood(x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a given image.

        :param x: the target images, assumed to be uint8 values rescaled to the range [-1, 1].
        :param means: the Gaussian mean tensor.
        :param log_scales: the Gaussian log stddev tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)

        # Calculate CDF at +/- 1/255
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = approx_standard_normal_cdf(min_in)

        # Compute log CDFs and probabilities
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min

        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs
    model.eval()
    with torch.no_grad():
        for x0, _ in data_loader:
            x0 = x0.to(device)
            batch_size = x0.size(0)
            num_samples += batch_size
            if num_samples <= 1000:
                num_run_samples = num_samples

                # 1. Prior KL (q(x_T | x_0) || p(x_T))
                noise = torch.randn_like(x0)
                xT = diffusion.forward_diffusion(x0, diffusion.T - 1)[0]
                prior_kl = 0.5 * torch.sum(xT ** 2, dim=[1, 2, 3])  # Isotropic Gaussian prior
                total_nll += prior_kl.sum()

                # 2. KL for each timestep (q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))
                for t in range(diffusion.T - 1, 0, -1):
                    xt = diffusion.forward_diffusion(x0, t)[0]
                    t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                    time_emb = time_embedding(t_tensor)

                    # Predict noise (epsilon_theta)
                    epsilon_pred = model(xt, time_emb)

                    # Compute mu_true based on Equation 7
                    beta_t = diffusion.beta[t].view(-1, 1, 1, 1).to(device)
                    alpha_t = 1 - beta_t
                    alpha_bar_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1).to(device)
                    alpha_bar_t_minus_1 = diffusion.alpha_bar[t - 1].view(-1, 1, 1, 1).to(device)

                    # Components for mu_true
                    x0_weight = (torch.sqrt(alpha_bar_t_minus_1) * beta_t) / (1 - alpha_bar_t) # self.posterior_mean_coef1 eq) 7
                    xt_weight = (torch.sqrt(alpha_t) * (1 - alpha_bar_t_minus_1)) / (1 - alpha_bar_t) # self.posterior_mean_coef2 eq) 7

                    # Compute mu_true
                    mu_true = x0_weight * x0 + xt_weight * xt # posterior mean

                    # Compute posterior variance and log variance

                    posterior_variance = beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t) # eq 7
                    posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20)) # clip to dont include 0


                    # Predicted posterior (mu_pred)
                    beta_t = diffusion.beta[t].view(-1, 1, 1, 1).to(device)
                    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1, 1).to(device)
                    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1).to(device)

                    mu_pred = (xt - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_pred) / sqrt_alpha_t # eq) 11

                    model_variance = posterior_variance
                    model_log_variance = posterior_log_variance_clipped

                    # Inside your NLL loop:
                    kl_t = gaussian_kl_divergence(
                        mu_true, posterior_log_variance_clipped,
                        mu_pred, model_log_variance
                    )
                    kl_t = kl_t.mean(dim=list(range(1, len(kl_t.shape))))
                    total_nll += kl_t.sum()/ np.log(2.0)

                # 3. Final log-likelihood (-log p_theta(x_0 | x_1))
                decoder_mean = mu_pred  # Predicted mean from the reverse process
                decoder_log_variance = posterior_log_variance_clipped  # Predicted log variance from the reverse process
                log_scales = 0.5 * decoder_log_variance  # Convert log variance to log stddev
                log_scales = log_scales.expand_as(x0)
                # Compute discretized Gaussian log likelihood
                decoder_nll = -discretized_gaussian_log_likelihood(
                    x=x0,  # Target image (original)
                    means=decoder_mean,
                    log_scales=0.5 * log_scales,  # Convert log variance to log stddev
                )
                assert decoder_nll.shape == x0.shape

                # Aggregate the NLL (convert to bits/dim)
                decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)
                print("decoder_nll: ", decoder_nll.sum())
                total_nll += decoder_nll.sum()
                print("total_nll: ", total_nll/num_run_samples)
                

    # Average NLL across all samples
    avg_nll = total_nll / num_run_samples
    return avg_nll

device = torch.device("cuda" if torch.cuda.is_available() else 
                    "mps" if torch.backends.mps.is_available() else 
                    "cpu")
print(f"Using Device: {device}")

# Step 1: Initialize the Sinusoidal Embeddings
time_embedding = SinusoidalPositionEmbeddings(total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512).to(device)

# Step 2: Initialize the U-Net and Diffusion
unet = UNet(input_channels=3, resolutions=[64, 128, 256, 512], time_emb_dims=512, dropout=0.1, use_attention=[False, True, False], heads=1).to(device)
diffusion = Diffusion(T=1000, beta_min=10e-5, beta_max=0.02, schedule='linear', device=device)

# Step 3: Load the trained model
model_path = "ddpm_CIFAR10_fina_advanced.pth"
print("Model_path: ", model_path)
saved = torch.load(model_path, map_location=device)

unet.load_state_dict(saved["model_state_dict"])
time_embedding.load_state_dict(saved["embedding_state_dict"])

train_data, _ = get_dataloader(dataset = "CIFAR10", batch_size = 10)

# Step 4: Generate samples
#samples, intermediates = sample_ddpm_with_intermediates(unet, diffusion, time_embedding, device, num_samples=10, dataset='MNIST')
print("Samples generated successfully!")

nll = compute_nll_vlb(unet, diffusion, train_data, time_embedding, device)
print(f"NLL: {nll}")
