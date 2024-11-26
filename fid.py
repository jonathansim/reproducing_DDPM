import numpy as np
import tensorflow as tf
from scipy.linalg import fractional_matrix_power
import torchvision.models as models
import torch
import tensorflow_hub as tfhub
from dataloader import get_dataloader_evaluation

'''
EXAMPLE MNIST:
# Get MNIST prediction model
MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)

dataloader = get_dataloader_evaluation(dataset = "MNIST", batch_size = 128)
mnist_real = get_mnist_real(dataloader, num_images=10000)
real_activations = compute_mnist_activations(mnist_real, mnist_classifier_fn)
generated_activations = compute_mnist_activations(samples, mnist_classifier_fn)

# Calculate FID
calculate_fid(real_activations, generated_activations)

'''

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
    # sum squared difference between means
    diff = np.sum((mu1 - mu2)**2)
 
    # calculate "geometric mean" of covariance matrices
    covmean = fractional_matrix_power(sigma1.dot(sigma2), 0.5)
    # check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_activations(tensors, num_batches, classifier_fn):
    """
    Given a tensor of of shape (batch_size, height, width, channels), computes
    the activiations given by classifier_fn.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    stack = tf.stack(tensors_list)
    activation = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(classifier_fn, stack, parallel_iterations=1, swap_memory=True),
    )
    return tf.concat(tf.unstack(activation), 0)


def get_mnist_real(dataloader, num_images=None):
    """
    Fetch a specified number of real images from the dataloader.
    
    Args:
        dataloader: PyTorch DataLoader providing the MNIST dataset.
        num_images: Number of images to fetch. If None, fetch all available images.
        
    Returns:
        real_images: A tensor containing the requested number of real images.
    """
    real_images = []

    for images, _ in dataloader:
        real_images.append(images)
        
        if num_images is not None and len(torch.cat(real_images, dim=0)) >= num_images:
            break
    
    real_images = torch.cat(real_images, dim=0)  # Shape: (num_real_images, channels, height, width)
    
    if num_images is not None:
        real_images = real_images[:num_images]
    
    return real_images


def compute_mnist_activations(samples, classifier_fn):
    """
    Compute activations for generated samples.
    Args:
        samples: Generated samples or real images (Tensor).
        classifier_fn: Classifier function to extract features.
    Returns:
        activations: Feature activations for generated images.
    """
    samples = torch.permute(samples, (0, 2, 3, 1))  # Change to (num_samples, height, width, channels)
    samples = tf.convert_to_tensor(samples.cpu().numpy())  # Convert to TensorFlow tensor

    # Compute activations
    activations = compute_activations(samples, num_batches=1, classifier_fn=classifier_fn)

    return activations.numpy()

def full_fid_mnist(generated_samples, num_images = 10000):
    """
    Computes the Fréchet Inception Distance (FID) score between generated samples
    and real MNIST images using a pre-trained MNIST classifier instead of Inceptionv3.

    Args:
        generated_samples (torch.Tensor): Tensor containing the generated samples 
            of shape (num_samples, channels, height, width).
        num_images (int, optional): Number of real MNIST images to use for FID computation. Should be the same as num_samples. 
            Defaults to 10,000.

    Returns:
        float: FID score between the generated and real MNIST images.
    """

    # Get MNIST prediction model
    MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
    mnist_classifier_fn = tfhub.load(MNIST_MODULE)

    dataloader = get_dataloader_evaluation(dataset = "MNIST", batch_size = 128)
    mnist_real = get_mnist_real(dataloader, num_images=num_images)
    real_activations = compute_mnist_activations(mnist_real, mnist_classifier_fn)
    generated_activations = compute_mnist_activations(generated_samples, mnist_classifier_fn)

    # Calculate FID
    fid = calculate_fid(real_activations, generated_activations)

    return fid


def compute_cifar_activations(samples, model):
    """
    Compute activations for samples using the CIFAR-10 feature extractor.
    Args:
        samples: Tensor of shape (num_samples, channels, height, width)
        model: Pre-trained model to extract features
    Returns:
        activations: Feature activations for input images
    """
    model.eval()
    samples = samples.to(next(model.parameters()).device)  # Move to the same device as the model
    with torch.no_grad():
        activations = model(samples)
    return activations.cpu().numpy()


def get_cifar_real_in_batches(dataloader, num_images=None):
    """
    Fetch CIFAR-10 images in smaller batches without concatenating them.
    Args:
        dataloader: PyTorch DataLoader providing the CIFAR-10 dataset.
        num_images: Number of images to fetch.
    Yields:
        Batches of real images from the CIFAR-10 dataset.
    """
    count = 0
    for images, _ in dataloader:
        count += len(images)
        if num_images and count >= num_images:
            img_need = num_images - count
            images = images[:img_need]
            yield images
            break

        # Yield images batch-by-batch
        yield images

        # Free memory
        del images
        torch.cuda.empty_cache()


def full_fid_cifar(generated_samples, num_images = 10000):
    """
    Computes the Fréchet Inception Distance (FID) between generated images and real CIFAR-10 images.
    Args:
        generated_samples (torch.Tensor): Tensor of generated samples 
            with shape (num_samples, channels, height, width).
        num_images (int, optional): Number of real CIFAR-10 images to use for FID computation. 
            Defaults to 10,000.
    Returns:
        float: FID score between the generated and real CIFAR-10 images.
    """

    # Load pre-trained InceptionV3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cifar_classifier = models.inception_v3(weights='DEFAULT', transform_input=False)
    cifar_classifier.fc = torch.nn.Identity()  # remove classification layer to extract features
    cifar_classifier = cifar_classifier.to(device)
    # Get CIFAR-10 train samples
    dataloader = get_dataloader_evaluation(dataset = "CIFAR10", batch_size = 100)
    real_image_activations = []

    for batch in get_cifar_real_in_batches(dataloader, num_images=num_images):
        activation = compute_cifar_activations(batch, cifar_classifier)
        real_image_activations.extend(activation)
    
    generated_image_activations = compute_cifar_activations(generated_samples, cifar_classifier)

    fid = calculate_fid(real_image_activations, generated_image_activations)

    return fid
