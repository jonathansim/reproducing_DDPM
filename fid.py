import numpy as np
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from scipy.linalg import fractional_matrix_power


# TODO Update to pytorch and our experiment setting

def calculate_fid(feat1, feat2):
    '''
    calculate fid between two sets of images
    feat1: feature vector obtained from inception model
    feat2: feature vector obtained from inception model
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


def preprocess(images, labels):
    """
    Preprocces helper function
    return: preprocessed image with pixel values scaled between -1 and 1
    """ 
    return tf.keras.applications.inception_v3.preprocess_input(images), labels


def get_features(images, model_weights, class_idx):
    '''
    Load Inceptionv3 model and get feature vectors of images
    images: Images to get feature representations for
    model_weights: File path to model weights or "imagenet" for imagenet weights
    class_idx: Dictionary that maps class_number (e.g. class 0 or class 1) to all the image indexes for this class
    return: feature vectors
    '''

    model = InceptionV3(input_shape=(299,299,3), include_top= False, weights=model_weights, pooling="avg")
    all_features = model.predict(images)

    features = {}
    for idx in class_idx:
       features[idx] = all_features[class_idx[idx]]
    
    return features