"""
Functions to build a variational autoencoder cnn

functions work so that number of Conv layers is variable
and the number of filters and latent dimensions

TODO - test different activations (z_mean/z_sigma?)
Q - is Conv/Max pooling architechture best? could we use padding instead?
TODO - allow for different reconstruction losses
TODO - add a check for input shape validity (doesn't go odd or to zero size)
Q - better way to stack layers? list seems real messy
Q - with padding='same', what effect does having deconv in decoder have instead of conv?
TODO - add channels capability, currently locked to 1 in decoder_model
"""
import numpy as np
import keras as k
import matplotlib.pyplot as plt
import os

from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.utils import plot_model, multi_gpu_model
from keras.datasets import mnist
from keras import optimizers

import tensorflow as tf

def encoder_model(input_data, layers, filters, latent_dims):
    """
    Initialise the Encoder part of the Variational Autoencoder
    Inputs:
    input_data - tensor, TFrecord for x_train
    layers - Int, Number of layers
    filters - list, filters for each layer
    latent_dims - Int, dims of the latent space
    Outputs:
    encoder - Encoder model ready to train
    inputs - Input layer
    outputs - list, [z_mean, z_sigma, z] layers
    """

    l = []
    l.append(Input(tensor=input_data, dtype=tf.float32)) #as we use TFrecords, need input layer

    for i in range(0,layers): #build convolution/pooling layers
        l.append(Conv2D(filters[i], (3,3), padding='same',
                        data_format='channels_last', name='Conv_'+str(i),
                        activation='relu')(l[i*2]))
        l.append(MaxPooling2D(pool_size=(2, 2),
                 data_format='channels_last',
                 name='Pool_'+str(i))(l[i*2+1]))

    conv_out = Flatten(name='Flatten_dims')(l[-1])

    # Z is essentially a prob distibution defined by mean and sigma
    # we sample z from the dist to pass into the decoder and fit the model
    z_mean = Dense(latent_dims, name='z_mean')(conv_out) #Q Do these need Activations?
    z_sigma = Dense(latent_dims, name='z_sigma')(conv_out)
    z = Lambda(VAE_sampling, output_shape=(latent_dims,), name='z')([z_mean,z_sigma])

    #Build the model, return the model, inputs and outputs
    encoder = Model(l[0], [z_mean,z_sigma,z], name='encoder')
    inputs = l[0]
    outputs = encoder(inputs)

    return encoder, inputs, outputs

def decoder_model(encoder, layers, filters, latent_dims):
    """
    Initialise the decoder part of the Variational Autoencoder
    Inputs:
        layers - Int, Number of layers
        filters - list, filters for each layer
        latent_dims - Int, dims of the latent space
    Outputs:
        decoder - decoder model ready to train
    """

    #first we need to know what shape the image was before/after flattening
    flat_dims = encoder.get_layer('Flatten_dims').output_shape
    pool_dims = encoder.get_layer('Flatten_dims').input_shape

    #input layer that takes in z, we use Dense and reshape to recover convolved image
    latent_inputs = Input(shape=(latent_dims,), name='z_sampling')
    latent_to_reshape = Dense(flat_dims[-1], activation='relu')(latent_inputs)
    reshape_to_up  = Reshape(pool_dims[1:])(latent_to_reshape)
 
    l = [reshape_to_up]

    for i in range(0,layers): #upsampling and deconv layers
        l.append(UpSampling2D(size=(2,2), data_format='channels_last',
                            name='Upsample_'+str(i))(l[i*2]))
        l.append(Conv2DTranspose(filters[-i-1], (3,3), padding='same',
                        data_format='channels_last', name='DeConv_'+str(i),
                        activation='relu')(l[i*2+1]))

    #final convolution layer recovers the original image
    l.append(Conv2D(1, (3,3), padding='same',
                    data_format='channels_last', name='decoder_output',
                    activation='sigmoid')(l[-1]))

    #build and return the model
    decoder = Model(latent_inputs, l[-1], name='decoder')

    return decoder

#take in the inputs and latent layers from encoder, the decoder, and use them
#to build the autoencoder model
def autoencoder_model(inputs, decoder, latent):
    autoencoder = Model(inputs = inputs, outputs = decoder(latent))
    return autoencoder

#output some summarys and save the models
def plot_models(self, filepath=None):
    self.autoencoder.summary()
    self.encoder.summary()
    self.decoder.summary()
    plot_model(self.autoencoder, to_file=filepath+'vae_autoencoder.png', show_shapes=True)
    plot_model(self.encoder, to_file=filepath+'vae_encoder.png', show_shapes=True)
    plot_model(self.decoder, to_file=filepath+'vae_decoder.png', show_shapes=True)

#custom loss function, actual loss in embedded as keras needs form loss(y_true,y_pred)
#by embedding that function, we can feed in the values for z and beta to find the 
#KL divergence
def vae_loss_func(z_mean, z_sigma, beta, loss=None):
#loss, str to pick loss function, defaults to binary crossentropy
    def recon(y_true, y_pred):
        if loss == 'mse':
            reconstruction_loss = mse(K.flatten(y_true),
                                                  K.flatten(y_pred))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(y_true),
                                                  K.flatten(y_pred))

        #KL divergence, we normalise by beta to tune amount of regularisation
        kl_loss = (1 + z_sigma -
                  K.square(z_mean) - 
                  K.exp(z_sigma))
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5 * beta

        return K.mean(reconstruction_loss+kl_loss)

    return recon

#sampling function for VAE
def VAE_sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

