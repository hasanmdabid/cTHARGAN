from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
# from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import LeakyReLU, UpSampling1D
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
import collections
from sklearn.preprocessing import MinMaxScaler
# from keras import initializers

import os.path
from pathlib import Path
from datetime import datetime

# Importing Plotting Library
# from matplotlib import pyplot as plt
# from keras.utils import plot_model
# import graphviz  # for showing model diagram

import sys
import warnings

# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Data manipulation
import numpy as np  # for data manipulation

print('numpy: %s' % np.__version__)  # print version

# Visualization
import matplotlib
import matplotlib.pyplot as plt  # for data visualizationa

print('matplotlib: %s' % matplotlib.__version__)  # print version
import graphviz  # for showing model diagram

print('graphviz: %s' % graphviz.__version__)  # print version

import sys
import warnings

# Disabling the Python Warning.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import math

# Load Sample Opportunity data


s1r1 = pd.read_csv('S1-ADL1.csv')
s1r2 = pd.read_csv('S1-ADL2.csv')
s1r3 = pd.read_csv('S1-ADL3.csv')
s1r4 = pd.read_csv('S1-ADL4.csv')
s1r5 = pd.read_csv('S1-ADL5.csv')
s1_drill = pd.read_csv('S1-Drill.csv')
s2r1 = pd.read_csv('S2-ADL1.csv')
s2r2 = pd.read_csv('S2-ADL2.csv')
s2r3 = pd.read_csv('S2-ADL3.csv')
s2r4 = pd.read_csv('S2-ADL4.csv')
s2r5 = pd.read_csv('S2-ADL5.csv')
s2_drill = pd.read_csv('S1-Drill.csv')
s3r1 = pd.read_csv('S3-ADL1.csv')
s3r2 = pd.read_csv('S3-ADL2.csv')
s3r3 = pd.read_csv('S3-ADL3.csv')
s3r4 = pd.read_csv('S3-ADL4.csv')
s3r5 = pd.read_csv('S3-ADL5.csv')
s3_drill = pd.read_csv('S3-Drill.csv')


def column_notation(data):
    data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18',
                    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                    '35',
                    '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
                    '52',
                    '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                    '69',
                    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                    '86',
                    '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101',
                    '102',
                    '103', '104', '105', '106', '107', 'Activity_Label']
    data['Activity_Label'] = data['Activity_Label'].replace(
        [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508,
         404508,
         408512, 407521, 405506], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    return data


def slided_numpy_array(data):
    import numpy as np
    x = data.to_numpy()

    # This function will generate the
    def get_strides(a, L, ov):
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])

        return np.array(out)

    L = 32  # Here L represent the Number of Samples of each DATA frame
    ov = 16  # ov represent the Sliding Window ration %%% Out of 32 the slided

    # print('After Overlapping')
    x = get_strides(x, L, ov)
    # print(x.shape)

    segment_idx = 0  # Index for the segment dimension
    nb_segments, nb_timestamps, nb_columns = x.shape
    data_to_save = np.zeros((nb_segments, nb_timestamps, nb_columns - 1), dtype=np.float32)
    labels_to_save = np.zeros(nb_segments, dtype=int)

    for i in range(0, nb_segments):
        labels = x[i][:][:]
        data_to_save[i] = labels[:, :-1]
        labels = x[i][:][:]
        labels = labels[:, -1]
        labels = labels.astype('int')  # Convert labels to int to avoid typing issues
        values, counts = np.unique(labels, return_counts=True)
        labels_to_save[i] = values[np.argmax(counts)]

    return data_to_save, labels_to_save


# Opportunity have 18 Classes Including the No activity lavel(0).

s1r1 = column_notation(s1r1)
s1r2 = column_notation(s1r2)
s1r3 = column_notation(s1r3)
s1r4 = column_notation(s1r4)
s1r5 = column_notation(s1r5)
s1_drill = column_notation(s1_drill)
s2r1 = column_notation(s2r1)
s2r2 = column_notation(s2r2)
s2r3 = column_notation(s2r3)
s2r4 = column_notation(s2r4)
s2r5 = column_notation(s2r5)
s2_drill = column_notation(s2_drill)
s3r1 = column_notation(s3r1)
s3r2 = column_notation(s3r2)
s3r3 = column_notation(s3r3)
s3r4 = column_notation(s3r4)
s3r5 = column_notation(s3r5)
s3_drill = column_notation(s3_drill)


def Numpy_array(x):
    df = x
    data_and_labels = df.to_numpy()
    np_data = data_and_labels[:, :-1]  # All columns except the last one
    labels = data_and_labels[:, -1]  # The last column
    labels = labels.astype('int')  # Convert labels to int to avoid typing issues

    nb_timestamps, nb_sensors = np_data.shape
    window_size = 32  # Size of the data segments
    timestamp_idx = 0  # Index along the timestamp dimension
    segment_idx = 0  # Index for the segment dimension

    # Initialise the result arrays
    nb_segments = int(math.floor(nb_timestamps / window_size))
    print('Starting segmentation with a window size of %d resulting in %d segments and number of features is %d  ...' %
          (window_size, nb_segments, nb_sensors))
    data_to_save = np.zeros((nb_segments, window_size, nb_sensors), dtype=np.float32)
    labels_to_save = np.zeros(nb_segments, dtype=int)
    print('Dimension and shape of the generated blank numpy array')

    while segment_idx < nb_segments:
        data_to_save[segment_idx] = np_data[timestamp_idx:timestamp_idx + window_size, :]
        # Check the majority label ocurring in the considered window
        current_labels = labels[timestamp_idx:timestamp_idx + window_size]
        values, counts = np.unique(current_labels, return_counts=True)
        labels_to_save[segment_idx] = values[np.argmax(counts)]
        timestamp_idx += window_size
        segment_idx += 1
    return data_to_save, labels_to_save


# Checking the Number of Labels in the Dataset
# print(s1r1['Activity_Label'].value_counts())

trainxs1r1, trainys1r1 = slided_numpy_array(s1r1)
trainxs1r2, trainys1r2 = slided_numpy_array(s1r2)
trainxs1r3, trainys1r3 = slided_numpy_array(s1r3)
trainxs1_drill, trainys1_drill = slided_numpy_array(s1_drill)
trainxs1r4, trainys1r4 = slided_numpy_array(s1r4)
trainxs1r5, trainys1r5 = slided_numpy_array(s1r5)
trainxs2r1, trainys2r1 = slided_numpy_array(s2r1)
trainxs2r2, trainys2r2 = slided_numpy_array(s2r2)
trainxs2_drill, trainys2_drill = slided_numpy_array(s2_drill)
trainxs3r1, trainys3r1 = slided_numpy_array(s3r1)
trainxs3r2, trainys3r2 = slided_numpy_array(s3r2)
trainxs3_drill, trainys3_drill = slided_numpy_array(s3_drill)

trainx = np.concatenate(
    (trainxs1r1, trainxs1r2, trainxs1r3, trainxs1_drill, trainxs1r4, trainxs1r5, trainxs2r1,
     trainxs2r2, trainxs2_drill, trainxs3r1, trainxs3r2, trainxs3_drill), axis=0)
trainy = np.concatenate(
    (trainys1r1, trainys1r2, trainys1r3, trainys1_drill, trainys1r4, trainys1r5, trainys2r1,
     trainys2r2, trainys2_drill, trainys3r1, trainys3r2, trainys3_drill), axis=0)

# ********************************************** Deleting the Majority Class*******************************************
"""
As we are not going to generate the majority class. So we dont need to train the GAN with the Majority class
"""


def minority_class_only(x, y):
    global trainx, trainy
    idx = np.where(y == 0)
    for i in idx:
        trainx = np.delete(x, [i], axis=0)
        trainy = np.delete(y, [i], axis=0)

    return trainx, trainy


trainx_min, trainy_min = minority_class_only(trainx, trainy)

trainy_min = trainy_min.reshape(len(trainy), 1)  # **********************************************Reshaping it to (row*1)
nr_samples, nr_rows, nr_columns = trainx_min.shape
print('Shape of trainx:', trainx_min.shape)
print('Shape of trainy:', trainy_min.shape)

unique, counts = np.unique(trainy_min, return_counts=True)
counter = collections.Counter(trainy)
print(counter)
plt.bar(counter.keys(), counter.values())
plt.savefig('Bar_plot_of_Class_Distribution', dpi=400)

########################################################################################################################
# Preparing the dataset to use for the CGAN
# Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
# Here I am Using The MinMaxScaler to normalize the data
# 1st I have reshaped the 3D trainx to 2D then perform the MinMaxsacler
# then I reshape it to 4D (Nr_samples, Nr_rows, Nr_columns, Nr_channels)
# Here channels = 1 (For Image channels = 3 for Colored image and 1 for the Gray scale Image)
# The dimension the of the Trainx is 4d (Sample, Row, column, channels) and labels is Column Vector (Samples, 1)

trainx_min = trainx_min.reshape(nr_samples * nr_rows, nr_columns)  # To sclae the data converting from 2D o 3D
scaler = MinMaxScaler(feature_range=(-1, 1))  # Declaring the scaler
scaler.fit(trainx_min)  # Fitting the scaler
trainx_min_scaled = scaler.transform(trainx_min)  # Transforming the DATA

# --------- Checking whether all the Feature value is scaled between +1 and -1 ------------------------------------------
max_val = np.max(trainx_min_scaled)
min_val = np.min(trainx_min_scaled)
print('Maximume_value after scaled:', max_val)
print('Minimume_value after scaled:', min_val)

X = trainx_min_scaled.reshape(nr_samples, nr_rows, nr_columns)  # Reshaping from 2D to 3D
print("Shape of the scaled array: ", X.shape)
dataset = [X, trainy_min]

print(dataset[0].shape)
print(dataset[1].shape)

"""
trainy = trainy.reshape(len(trainy), 1)  #*****************************************************Reshaping it to (row*1)
nr_samples, nr_rows, nr_columns = trainx.shape
print('Shape of trainx:', trainx.shape)
print('Shape of trainy:', trainy.shape)

########################################################################################################################
# Preparing the dataset to use for the CGAN
# Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
# Here I am Using The MinMaxScaler to normalize the data
# 1st I have reshaped the 3D trainx to 2D then perform the MinMaxsacler
# then I reshape it to 4D (Nr_samples, Nr_rows, Nr_columns, Nr_channels)
# Here channels = 1 (For Image channels = 3 for Colored image and 1 for the Gray scale Image)
# The dimension the of the Trainx is 4d (Sample, Row, column, channels) and labels is Column Vector (Samples, 1)
########################################################################################################################

trainx_1D = trainx.reshape(nr_samples*nr_rows, nr_columns)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(trainx_1D)
trainx_1D_scaled = scaler.transform(trainx_1D)


##------- Checking whether all the Feature value is scaled between +1 and -1 --------------------------------------
max_val = np.max(trainx_1D_scaled)
min_val = np.min(trainx_1D_scaled)
print('Maximume_value after scaled:', max_val)
print('Minimume_value after scaled:', min_val)

X = trainx_1D_scaled.reshape(nr_samples, nr_rows, nr_columns)
print("Shape of the scaled array: ", X.shape)
dataset = [X, trainy]
print(dataset[0].shape)
print(dataset[1].shape)
# values, counts = np.unique(trainy, return_counts=True)
"""


##################################################################################################################
# Define generator, discriminator, gan and other helper functions
# We will use functional way of defining model as we have multiple inputs;
# both images and corresponding labels.
##################################################################################################################

# define the standalone discriminator model
# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.

# Unlike regular GAN here we are also providing number of classes as input.
# Input to the model will be both images and labels.

# define the standalone discriminator model
def define_discriminator(in_shape=(32, 107), n_classes=18):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    # image input
    # Label Inputs

    in_label = Input(shape=(1,), name='Discriminator-Label-Input-Layer')  # Input Layer
    lbls = Embedding(n_classes, 50, name='Discriminator-Label-Embedding-Layer')(in_label)  # Embed label to vector

    # Scale up to image dimensions
    n_nodes = in_shape[0] * in_shape[1]
    lbls = Dense(n_nodes, name='Discriminator-Label-Dense-Layer')(lbls)
    lbls = Reshape((in_shape[0], in_shape[1]), name='Discriminator-Label-Reshape-Layer')(lbls)  # New shape

    # Image Inputs
    in_image = Input(shape=in_shape, name='Discriminator-Image-Input-Layer')

    # Combine both inputs so it has two channels
    concat = Concatenate(name='Discriminator-Combine-Layer')([in_image, lbls])

    # downsample to 14x14
    dis = Conv1D(16, 3, strides=2, padding='same')(concat)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.2)(dis)
    # normal
    dis = Conv1D(32, 3, strides=2, padding='same')(dis)
    # dis = BatchNormalization()(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.2)(dis)
    # downsample to 7x7
    dis = Conv1D(64, 3, strides=2, padding='same')(dis)
    # dis = BatchNormalization()(dis)
    dis = LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.2)(dis)

    # flatten feature maps
    dis = Flatten()(dis)
    # real/fake output

    output_layer = Dense(1, activation='sigmoid', name='Discriminator-Output-Layer')(dis)  # Output Layer

    # Define model
    model = Model([in_image, in_label], output_layer, name='Discriminator')

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=18):
    # label input
    in_label = Input(shape=(1,), name='Generator-Label-Input-Layer')  # Input of dimension 1
    # embedding for categorical input
    # each label (total 18 classes for Opportunity), will be represented by a vector of size 50.
    li = Embedding(n_classes, 50, name='Generator-Label-Embedding-Layer')(in_label)  # Shape 1,50

    # linear multiplication
    n_nodes = 8 * 27  # To match the dimensions for concatenation later in this step.
    li = Dense(n_nodes, name='Generator-Label-Dense-Layer')(li)  # 1,64
    # reshape to additional channel
    li = Reshape((8, 27), name='Generator-Label-Reshape-Layer')(li)

    # image generator input
    in_lat = Input(shape=(latent_dim,), name='Generator-Latent-Input-Layer')  # Input of dimension 100

    # foundation for 8x8 image
    # We will reshape input latent vector into 8x8 image as a starting point.
    # So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output
    # it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    # Note that this part is same as unconditional GAN until the output layer.
    # While defining model inputs we will combine input label and the latent input.

    n_nodes = 8 * 27  # This Part is very Important

    gen = Dense(n_nodes, name='Generator-Foundation-Layer')(in_lat)  # shape=8192
    gen = LeakyReLU(alpha=0.2, name='Generator-Foundation-Layer-Activation-1')(gen)
    gen = Reshape((8, 27), name='Generator-Foundation-Layer-Reshape-1')(gen)  # Shape=8x8x32

    # merge image gen and label input
    merge = Concatenate(name='Generator-Combine-Layer')(
        [gen, li])  # Shape=8x32x108 (Extra channel corresponds to the label)

    # up-sample to 32x32 ========== HIDDEN layer 1
    gen = LSTM(54, return_sequences=True, name='Generator-Hidden-Layer-1')(merge)  # 16x107x64
    # gen = BatchNormalization(momentum=0.5)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-1')(gen)
    gen = UpSampling1D(size=2)(gen)

    gen = LSTM(108, return_sequences=True, name='Generator-Hidden-Layer-2')(gen)  # 16x107x64
    # gen = BatchNormalization(momentum=0.5)(gen)
    gen = LeakyReLU(alpha=0.2, name='Generator-Hidden-Layer-Activation-2')(gen)
    gen = UpSampling1D(size=2)(gen)

    # output
    out_layer = Conv1D(filters=107, kernel_size=3, activation='tanh', padding='same', name='Generator-Output-Layer')(
        gen)  # 32x107
    # define model
    model = Model([in_lat, in_label], out_layer, name='Generator')
    return model


# #Generator is trained via GAN combined model.
# define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately so here only generator will be trained by keeping
# the discriminator constant.
def define_gan(g_model, d_model):
    d_model.trainable = False  # Discriminator is trained separately. So set to not trainable.

    ## connect generator and discriminator...
    # first, get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input  # Latent vector size and label size
    # get image output from the generator model
    gen_output = g_model.output  # 32x32x3

    # generator image output and corresponding input label are inputs to discriminator
    gan_output = d_model([gen_output, gen_label])

    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# # select real samples
# pick a batch of random real samples to train the GAN
# In fact, we will train the GAN on a half batch of real images and another
# half batch of fake images.
# For each real image we assign a label 1 and for fake we assign label 0.
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples, 1))  # Label=1 indicating they are real
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=18):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    labels = labels.reshape(len(labels), 1)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
# Supply the generator, latent_dim and number of samples as input.
# Use the above latent point generator to generate latent points.
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))  # Label=0 indicating they are fake
    return [images, labels_input], y


def saveResultsCSV(gen_activation, epochs, batch_per_epoch, d_loss_real, d_loss_fake, g_loss):
    path = './results/'
    if not os.path.exists(path):
        os.makedirs(path)

    fileString = './results/D_and_G_model_Loss_1K_64Batch_minorityonly_conv1d_lstm.csv'
    file = Path(fileString)
    now = datetime.now()

    if not file.exists():
        f = open(fileString, 'w')
        f.write('gen_activation; epochs; batch_per_epoch; d_loss_real; d_loss_fake; g_loss\n')
        f.close()
    with open(fileString, "a") as f:
        f.write('{};{};{};{};{};{}\n'.format(gen_activation, epochs, batch_per_epoch, d_loss_real, d_loss_fake, g_loss))
    f.close()


# train the generator and discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images from our true/real dataset.
# Then, generating a set of images using the generator.
# Feed both set of images into the Discriminator.
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss.

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)  # the discriminator model is updated for a half batch of real samples
    # and a half batch of fake samples, combined a single batch.
    # manually enumerate epochs
    gen_activation = 'tanh'

    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # Train the discriminator on real and fake images, separately (half batch each)
            # Research showed that separate training is more effective.
            # get randomly selected 'real' samples
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection
            # of samples you provide
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)

            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            # This is where the generator is trying to trick discriminator into believing
            # the generated image is true (hence value of 1 for y)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # Generator is part of combined model where it got directly linked with the discriminator
            # Train the generator with latent_dim as x and 1 as y.
            # Again, 1 as the output as it is adversarial and if generator did a great
            # job of folling the discriminator then the output would be 1 (true)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # Print losses on this batch
            print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' % (
                i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
            saveResultsCSV(gen_activation, i, j, d_loss_real, d_loss_fake, g_loss)
            # if (i % 2500) == 0:
            #    g_model.save(f'CGAN_OPPO_{i}epochs.h5')

    # save the generator model
    g_model.save('CGAN_CONV1D_LSTM_OPPO_1k_epochs_64Batch_minorityonly.h5')


# Train the GAN

# size of the latent space, number of Batch and number of Epochs

# ********************************************************************
latent_dim = 150  # *
n_batch = 64  # *
half_batch = 32  # *
n_epochs = 100  # *
# ********************************************************************

# create the discriminator
d_model = define_discriminator()

# create the generator
g_model = define_generator(latent_dim)

# create the gan
gan_model = define_gan(g_model, d_model)
gan_model.summary()
# Train Model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch)

# **************************->Deleting all the vriables<-*********************************************
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name]
