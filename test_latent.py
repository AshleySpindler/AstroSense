from Autoencoder import Models_TFrecord as Models
import Data.GalZoo2_TFIngest as Data_In

import keras.backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np
import imageio

"""
Run script for testing latent encodings
"""

train_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'

x_train = Data_In.create_dataset(train_data, 200, None)

"""
Hyperparameters:
    layers = number of convolution/deconv layers in the encoder and decoder
    filters = number of convolutional filters in each layer
    latent = dimensionality of the latent space
    beta = weighting factor for Kullback–Leibler divergence (see B-VAE for details)
    B-VAE - https://openreview.net/forum?id=Sy2fzU9gl
    lr = learning rate
"""

layers = 3
filters = [9,5,47]
latent = 44
beta = 0
#beta = beta * latent / x_shape[1]**2 #we need to normalise beta, because it's very dependent on dimensionality
lr = 0.0008317883360924015

#make encoder, pull out pointers to the input layer and latent space
encoder, inputs, latent_z = Models.encoder_model(x_train, layers, filters, latent)

#make decoder
decoder = Models.decoder_model(encoder, layers, filters, latent)

#this looks weird, it's calling a function in Models that sets output as decoder(latent_z[2])
vae = Models.autoencoder_model(inputs, decoder, latent_z[2])

vae.compile(optimizers.Adam(lr=lr),
            loss=Models.vae_loss_func(latent_z[0],
                               latent_z[1], beta, loss='mse'),
            target_tensors=inputs)

#load weights for the model
vae.load_weights('Data/Scratch/GalZoo2/bayes_Best_Weights.h5') #save the trained weights

test_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'
x_test = Data_In.create_dataset(test_data, 200, None)

with tf.Session() as sess:
    img = sess.run(x_test)

encoded_img = encoder.predict(x_test, steps=1)

decoded_img = vae.predict(x_test, steps=1)

z = encoded_img[2][10]
z_range = np.linspace(-3,3,5)

f, ax = plt.subplots(10,3)
for i in range(0,10):
    ax[i,0].imshow(img[i,:,:,0], vmin=0, vmax=1)
    ax[i,2].imshow(decoded_img[i,:,:,0], vmin=0, vmax=1)
    ax[i,1].imshow(encoded_img[2][i].reshape(11,4))

ax[0,0].set_title('Greyscale Image')
ax[0,1].set_title('Encoded Image')
ax[0,2].set_title('Reconstructed Image')

f, ax = plt.subplots(5,44)
for i in range(0,44):
    images = []
    for j in range(0,5):
        z_n = z.copy()
        z_n[i] = z_range[j]
        new_img = decoder.predict(z_n.reshape(1,44))
        ax[j,i].imshow(new_img[0,:,:,0])
        #plt.savefig('Data/Scratch/GalZoo2/imgs/z-'+str(i)+'-step-'+str(j)+'.png')
        #plt.close()
        images.append(new_img[0,:,:,0])
    imageio.mimsave('Data/Scratch/GalZoo2/imgs/bayes_z-'+str(i)+'.gif', images)



f, ax = plt.subplots(1,20)
z_range = np.linspace(-3,3,20)
for j in range(0,20):
    z_n = z.copy()
    z_n[1] = z_range[j]
    new_img = decoder.predict(z_n.reshape(1,44))
    ax[j].imshow(new_img[0,:,:,0])

plt.show()

