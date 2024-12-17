import numpy as np
from scipy.linalg import fractional_matrix_power
import torchvision.models as models
import torch
from dataloader import get_dataloader_evaluation
import torchvision


def calculate_fid(feat1, feat2):
    '''
    calculate fid between two sets of images
    feat1: real images - feature vector obtained from inception model (or alternative model)
    feat2: generated images - feature vector obtained from inception model (or alternative model)
    return: fid score
    '''
    # calculate mean and covariance of features
    mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False) 
    mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
    print("Mean1: ", np.shape(mu1))
    print("Mean2: ", np.shape(mu2))
    # sum squared difference between means
    diff = np.sum((mu1 - mu2)**2)
    print("diff: ", diff)
    # calculate "geometric mean" of covariance matrices
    covmean = fractional_matrix_power(sigma1.dot(sigma2), 0.5)
    # check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid




def compute_activations(samples, model, batched = False, batch_size = 100):
    """
    Compute activations from samples based on model.
    Args:
        samples: Tensor of shape (num_samples, channels, height, width)
        model: Pre-trained model to extract features
    Returns:
        activations: Feature activations for input images
    """
    model.eval()
    samples = samples.to(next(model.parameters()).device)  # Move to the same device as the model

    if batched:
        activations = []
        with torch.no_grad():
            for i in range(0, samples.size(0), batch_size):
                batch = samples[i:i+batch_size]
                batch_activations = model(batch).cpu().numpy()
                activations.append(batch_activations)
        return np.concatenate(activations, axis=0)
    else:
        with torch.no_grad():
            activations = model(samples)
        return activations.cpu().numpy()


def get_images_in_batches(dataloader, num_images=None):
    """
    Fetch e.g. CIFAR-10 images in smaller batches without concatenating them.
    Args:
        dataloader: PyTorch DataLoader providing the dataset.
        num_images: Number of images to fetch.
    Yields:
        Batches of real images from the dataset.
    """
    count = 0
    for images, _ in dataloader:
        count += len(images)
        if num_images and count >= num_images:
            yield images
            break

        # Yield images batch-by-batch
        yield images

        # Free memory
        del images
        torch.cuda.empty_cache()


import torch
from torch.cuda.amp import autocast

def full_fid(generated_samples, data, num_images=10000, chunk_size=256):
    """
    Computes the Fréchet Inception Distance (FID) between generated images and real CIFAR-10/MNIST images.
    Args:
        generated_samples (torch.Tensor): Tensor of generated samples.
        data (str): MNIST or CIFAR10.
        num_images (int, optional): Number of real images to use. Defaults to 10,000.
        chunk_size (int): Number of images processed at a time to save memory.
    Returns:
        float: FID score.
    """

    # Load pre-trained InceptionV3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_classifier = models.inception_v3(weights='DEFAULT', transform_input=False)
    inception_classifier.fc = torch.nn.Identity()
    inception_classifier = inception_classifier.to(device).eval()

    # Get real image activations
    dataloader = get_dataloader_evaluation(dataset=data, batch_size=50)
    print("DATA: ", data)
    real_image_activations = []
    for batch in get_images_in_batches(dataloader, num_images=num_images):
        with torch.no_grad(), autocast():  # Mixed precision
            activation = compute_activations(batch.to(device), inception_classifier)
        real_image_activations.extend(activation)
    real_image_activations = real_image_activations[:num_images]

    # Normalize and compute activations for generated images
    min_val, max_val = generated_samples.min(), generated_samples.max()
    resized_samples = (generated_samples - min_val) / (max_val - min_val)  # Rescale to [0, 1]
    resized_samples = resized_samples.repeat(1, 3, 1, 1)  # Match RGB channels
    resized_samples = torchvision.transforms.functional.resize(resized_samples, size=(299, 299))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)

    generated_image_activations = []
    for i in range(0, resized_samples.size(0), chunk_size):
        chunk = resized_samples[i:i + chunk_size].to(device)
        with torch.no_grad(), autocast():  # Mixed precision
            chunk = torchvision.transforms.functional.normalize(chunk, mean=mean, std=std)
            activation = compute_activations(chunk, inception_classifier)
        generated_image_activations.extend(activation)
        del chunk  # Free memory
        torch.cuda.empty_cache()

    print(np.shape(generated_image_activations))
    print(np.shape(real_image_activations))
    
    # Calculate FID
    fid = calculate_fid(real_image_activations, generated_image_activations)
    return fid


def get_inception_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_classifier = models.inception_v3(weights='DEFAULT', transform_input=False)
    inception_classifier.fc = torch.nn.Identity()  # remove classification layer to extract features
    inception_classifier = inception_classifier.to(device)
    return inception_classifier

def get_real_image_activations(inception_classifier, data, num_images = 10000):
    # Get train samples
    dataloader = get_dataloader_evaluation(dataset = data, batch_size = 100)
    print("DATA: ", data)
    # Get activations for real images
    real_image_activations = []
    for batch in get_images_in_batches(dataloader, num_images=num_images):
        activation = compute_activations(batch, inception_classifier)
        real_image_activations.extend(activation)
    real_image_activations = real_image_activations[:num_images]
    return real_image_activations

def calculate_fid_generated_samples(generated_samples, real_image_activations, inception_classifier,device, num_images = 10000):
        # Get activations of generated images
    min_val = generated_samples.min()
    max_val = generated_samples.max()
    resized_samples = (generated_samples - min_val) / (max_val - min_val) # rescale pixel values to [0,1]
    print("Rescale generated images to [0,1]")
    resized_samples = resized_samples.repeat(1, 3, 1, 1)
    resized_samples = torchvision.transforms.functional.resize(resized_samples, size = (299, 299)) # resize for inception
    resized_samples = torchvision.transforms.functional.normalize(resized_samples, 
        mean=torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device), 
        std=torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
    )
    generated_image_activations = compute_activations(resized_samples, inception_classifier, batched = True)
    print(np.shape(generated_image_activations))
    print(np.shape(real_image_activations))
    fid = calculate_fid(real_image_activations, generated_image_activations)
    return fid