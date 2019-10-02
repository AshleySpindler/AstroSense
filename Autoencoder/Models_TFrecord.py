import numpy as np
import keras as k
import matplotlib.pyplot as plt
import os

from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.utils import plot_model, multi_gpu_model
from keras.datasets import mnist
from keras import optimizers

import tensorflow as tf

    #TODO build encoder model, variable n. of layers, mean and sigma latent space
def encoder_model(input_data, layers, filters, latent_dims):
    """
    Initialise the Encoder part of the Variational Autoencoder
    Inputs:
    input_shape - Arraylike, shape of input image
    layers - Int, Number of layers
    filters - Arraylike, filters for each layer
    latent_dims - Int, dims of the latent space
    Outputs:
    encoder - Encoder model ready to train
    """

    l = []
    l.append(Input(tensor=input_data, dtype=tf.float32))
        
    for i in range(0,layers):
        l.append(Conv2D(filters[i], (3,3), padding='same',
                        data_format='channels_last', name='Conv_'+str(i),
                        activation='relu')(l[i*2]))
        l.append(MaxPooling2D(pool_size=(2, 2),
                 data_format='channels_last',
                 name='Pool_'+str(i))(l[i*2+1]))
    conv_out = Flatten(name='Flatten_dims')(l[-1])
    z_mean = Dense(latent_dims, name='z_mean')(conv_out)
    z_sigma = Dense(latent_dims, name='z_sigma')(conv_out)
    z = Lambda(VAE_sampling, output_shape=(latent_dims,), name='z')([z_mean,z_sigma])

    encoder = Model(l[0], [z_mean,z_sigma,z], name='encoder')
    inputs = l[0]
    outputs = encoder(inputs)

    encoder.summary()

    return encoder, inputs, outputs

def decoder_model(encoder, layers, filters, latent_dims):
    """
    Initialise the decoder part of the Variational Autoencoder
    Inputs:
        input_shape - Arraylike, shape of input image
        layers - Int, Number of layers
        filters - Arraylike, filters for each layer
        latent_dims - Int, dims of the latent space
    Outputs:
        decoder - decoder model ready to train
    """

    flat_dims = encoder.get_layer('Flatten_dims').output_shape
    pool_dims = encoder.get_layer('Flatten_dims').input_shape

    latent_inputs = Input(shape=(latent_dims,), name='z_sampling')
    latent_to_reshape = Dense(flat_dims[-1], activation='relu')(latent_inputs)
    reshape_to_up  = Reshape(pool_dims[1:])(latent_to_reshape)
 
    l = [reshape_to_up]

    for i in range(0,layers):
        l.append(UpSampling2D(size=(2,2), data_format='channels_last',
                            name='Upsample_'+str(i))(l[i*2]))
        l.append(Conv2D(filters[-i-1], (3,3), padding='same',
                        data_format='channels_last', name='DeConv_'+str(i),
                        activation='relu')(l[i*2+1]))
    l.append(Conv2D(1, (3,3), padding='same',
                    data_format='channels_last', name='decoder_output',
                    activation='sigmoid')(l[-1]))
    decoder = Model(latent_inputs, l[-1], name='decoder')
    decoder.summary()
    return decoder

def autoencoder_model(inputs, decoder, latent):
    autoencoder = Model(inputs = inputs, outputs = decoder(latent))
    return autoencoder

def plot_models(self):
    self.autoencoder.summary()
    self.encoder.summary()
    self.decoder.summary()
    plot_model(self.autoencoder, to_file='vae_autoencoder.png', show_shapes=True)
    plot_model(self.encoder, to_file='vae_encoder.png', show_shapes=True)
    plot_model(self.decoder, to_file='vae_decoder.png', show_shapes=True)

def vae_loss_func(z_mean, z_sigma, beta):

    def recon(y_true, y_pred):
        reconstruction_loss = binary_crossentropy(K.flatten(y_true),
                                                  K.flatten(y_pred))

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

