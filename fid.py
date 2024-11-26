import numpy as np
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from scipy.linalg import fractional_matrix_power
from torchvision import datasets, transforms
from torch.utils import data
import torch
import tensorflow_hub as tfhub
from dataloader import get_dataloader

'''
EXAMPLE:

dataloader, _ = get_dataloader(dataset = "MNIST", batch_size = 128)
mnist_real = get_mnist_real(dataloader, num_images=10000)
real_activations = compute_mnist_activations(mnist_real, mnist_classifier_fn)

calculate_fid(activations[5000:10000], activations[:5000])

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

# Get MNIST prediction model
MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)

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


