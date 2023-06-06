##########################################################
# Now, let us load the generator model and generate images
# Lod the trained model and generate a few images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
import tensorflow as tf

#---------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

from matplotlib import pyplot as plt
# Note: CIFAR10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=17):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]
#-----------------------------------------------------------------------------------------------------------------------
# load  Cifar Model model
model_cifar = load_model('cifar_conditional_generator_250epochs.h5')
# generate multiple images
latent_points_cifar, _ = generate_latent_points(100, 100)  # Input (Latent Point Dimension, n_Samples) .
labels_cifar = asarray([x for _ in range(10) for x in range(10)])
print("Shape of Cifar Latent point:", latent_points_cifar.shape)
print('Shape of Cifar labels:', labels_cifar.shape)
# Generate CIFAR data
X_cifar = model_cifar.predict([latent_points_cifar, labels_cifar])
# scale from [-1,1] to [0,1]
X_cifar = (X_cifar + 1) / 2.0
X_cifar = (X_cifar * 255).astype(np.uint8)
print('Shape of Cifar Generated DATA:', X_cifar.shape)  # This will Generate 100 SAMPLES of (10 images of each 10 classes)


#----------------------------------------------HAR OPPO-----------------------------------------------------------------
# Load HAR-OPPO model
model_har = load_model('CGAN_500_epochs_32Batch_Minority_withoutBatchNormalization_labelSmoothing_conv2d_lstm.h5')
latent_points_har, _ = generate_latent_points(100, 34000)  # Input (Latent Point Dimension, n_Samples) .
# specify labels - generate 10 sets of labels each gping from 0 to 9
labels_har = asarray([x for _ in range(1, 2001) for x in range(1, 18)])  # Dimension of Labels should be same as N_samples.
print("Shape of Har Latent point:", latent_points_har.shape)
print("Shape of Har Labels:", labels_har.shape)
print(labels_har)
from joblib import load
scaler = load('minmax_scaler.bin')

# Generate Har Data
X_har = model_har.predict([latent_points_har, labels_har])
print('Shape of HAR generated data', X_har.shape)

max_val_sacled = np.max(X_har)
min_val_scaled = np.min(X_har)
print('Maximume_value Before rescaled:', max_val_sacled)
print('Minimume_value Before rescaled:', min_val_scaled)

nr_samples, nr_rows, nr_columns, nr_channels = X_har.shape
#Rescale from [-1, 1] by using the MinMax scaler inverse transform
X_har = X_har.reshape(nr_samples * nr_rows, nr_columns)
#Rescale from [-1, 1] by using the MinMax scaler inverse transform
X_har = scaler.inverse_transform(X_har)
X_har = X_har.reshape(nr_samples, nr_rows, nr_columns, nr_channels)
print('After rescalling and reshape of HAR generated data', X_har.shape)

max_val = np.max(X_har)
min_val = np.min(X_har)
print('Maximume_value after scaled:', max_val)
print('Minimume_value after scaled:', min_val)
print(labels_har.shape)

#-----------------------------------------------Checking the Labels ratio-----------------------------------------------

import collections
unique, counts = np.unique(labels_har, return_counts=True)
counter = collections.Counter(labels_har)
print(counter)
plt.bar(counter.keys(), counter.values())
plt.savefig('Bar_plot_of_Class_Distribution_of_Synthetic_Data', dpi=400)

#---------------------------------------------Checking the Maximume and Minimume value----------------------------------




