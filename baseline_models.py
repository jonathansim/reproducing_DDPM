import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from fid import full_fid
from inception_score import calculate_inception_score
import wandb
import random

def compute_pixel_distribution(dataloader):
    """
    Compute mean and std for each pixel position in the dataset.
    Args:
        dataloader: DataLoader containing the dataset.
    Returns:
        mean: Pixel-wise mean (height, width) or (channels, height, width).
        std: Pixel-wise std deviation (same shape as mean).
    """
    pixel_sum = 0
    pixel_sum_squared = 0
    num_samples = 0

    for images, _ in dataloader:
        pixel_sum += images.sum(dim=0)  # Sum across batch
        pixel_sum_squared += (images ** 2).sum(dim=0)
        num_samples += images.size(0)  # Number of samples in batch

    mean = pixel_sum / num_samples
    std = torch.sqrt(pixel_sum_squared / num_samples - mean ** 2)
    return mean, std

def generate_images(mean, std, num_images=10):
    """
    Generate images by sampling from the learned distribution.
    Args:
        mean: Pixel-wise mean (height, width) or (channels, height, width).
        std: Pixel-wise std deviation (same shape as mean).
        num_images: Number of images to generate.
    Returns:
        Tensor of generated images.
    """
    return torch.normal(mean=mean, std=std).unsqueeze(0).repeat(num_images, 1, 1, 1)

# make new dataloaders without DDPM specific transforms
def get_mnist_dataloader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./temp", train=True, download=True, transform=transform)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True)

def get_cifar10_dataloader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root="./temp", train=True, download=True, transform=transform)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True)



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
    mode_for_wandb = "online"
    run_name = "Baseline_pixel_average"
    wandb.init(project='ddpm', entity='dl_ddpm', mode=mode_for_wandb, name=run_name)
    

    # MNIST
    data = "MNIST"
    print(data)
    batch_size = 128
    dataloader = get_mnist_dataloader(batch_size=batch_size)
    mean, std = compute_pixel_distribution(dataloader)
    generated_images = generate_images(mean, std, num_images=10000)
    generated_images = generated_images.to(device)
    wandb.log({f"Generated Samples {data}": [wandb.Image(generated_images[0], caption=f"Generated Image: {data}")]})
    fid = full_fid(generated_samples=generated_images, data=data, num_images = 10000)
    print("Fid: ", fid)
    wandb.log({"model": data})   
    wandb.log({"FID1": fid, "model": data})


    # CIFAR
    data = "CIFAR10"
    print(data)
    dataloader = get_cifar10_dataloader(batch_size=batch_size)
    mean, std = compute_pixel_distribution(dataloader)
    generated_images = generate_images(mean, std, num_images=10000)
    generated_images = generated_images.to(device)
    wandb.log({f"Generated Samples {data}": [wandb.Image(generated_images[0], caption=f"Generated Image: {data}")]})
    fid = full_fid(generated_samples=generated_images, data = data, num_images = 10000)
    print("Fid: ", fid)   
    wandb.log({"model": data})
    wandb.log({"FID2": fid, "model": data})

    mean_is, std_is = calculate_inception_score(samples=generated_images, dataset=data, device=device, batch_size=32, splits=10)
    print(f"Inception Score: {mean_is} ± {std_is}")
    wandb.log({"IS": mean_is, "model": data})
    wandb.finish()


    visualize = False
    if visualize:
        # Visualize generated images
        grid = torchvision.utils.make_grid(generated_images, nrow=5, normalize=True)
        plt.figure(figsize=(10, 5))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.title("Generated Images")
        plt.axis("off")
        plt.show()
