from Autoencoder import Models_TFrecord as Models
import Data.GalZoo2_TFIngest as Data_In

import keras.backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import optimizers

import matplotlib.pyplot as plt

"""
Run script for fitting a variational autoencoder to Galaxy Zoo 2 Images

TODO - Fix Multi-gpu implementation, input tensors getting lost when model
       is copied
TODO - standardise input data so we can check GZ2 morphologies
TODO - fix model testing and reconstructions so it picks out same galaxies
TODO - test optimisers
"""

epochs = 100 #number of epochs to train
batch_size = 200 #batch size (limited by GPU memory)
n_inputs = 10000 #number of objects in x_train
steps_per_epoch = n_inputs//batch_size #tells keras how many bacthes to do
shuffle_size = 100 #how many objects to shuffle (not used right now)

#Load in training and validation data from TFrecords
train_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords' 
valid_data = 'Data/Scratch/GalZoo2/galzoo_valid.tfrecords'

x_train = Data_In.create_dataset(train_data, batch_size, None)
x_valid = Data_In.create_dataset(valid_data, batch_size, None)

x_shape = x_train.get_shape().as_list()

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
filters = [32,32,64]
latent = 32
beta = 1
beta = beta * latent / x_shape[1]**2 #we need to normalise beta, because it's very dependent on dimensionality
lr = 1e-3

#make encoder, pull out pointers to the input layer and latent space
encoder, inputs, latent_z = Models.encoder_model(x_train, 3, filters, latent)

#make decoder
decoder = Models.decoder_model(encoder, layers, filters, latent)

#this looks weird, it's calling a function in Models that sets output as decoder(latent_z[2])
vae = Models.autoencoder_model(inputs, decoder, latent_z[2])

#I dunno whats going on here, when you make it a multi-gpu-model it seems to lose
#its connection to x_train, left out for now (it trains pretty quick on 1 V100 tho)

#try:
#    vae = multi_gpu_model(vae, gpus=2, cpu_relocation=True)
#    print("Training using multiple GPUs..")
#except ValueError:
#    vae = vae
#    print("Training using single GPU or CPU..")

#compile the model, we make a call to a custom loss function
vae.compile(optimizers.Adam(lr=lr),
            loss=Models.vae_loss_func(latent_z[0],
                               latent_z[1], beta),
            target_tensors=inputs)

#fit the model, if x_valid is different size to x_train you need to change steps_per_epoch
vae.fit(batch_size=None, epochs=epochs, steps_per_epoch=steps_per_epoch,
        validation_data=(x_valid, None), validation_steps=steps_per_epoch)
vae.save_weights('Data/Scratch/GalZoo2/vae_cnn_test.h5') #save the trained weights

"""
Plot some reconstructions of test data

this bit isn't working right now, in theory we want to take the input,
grab 10-20 galaixes, visualise the latent spaces and reconstructions
because of how TFrecords are accessed, each call brings out a different
set of galaxies
"""
test_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'
x_test = Data_In.create_dataset(test_data, batch_size, None)
steps = 100000//batch_size

with tf.Session() as sess:
    img = sess.run(x_test)[0:10]

decoded_img = vae.predict(x_test, steps=steps)[0:10]
encoded_img = encoder.predict(x_test, steps=steps)[0:10]

f, ax = plt.subplots(10,3)
for i in range(0,10):
    ax[i,0].imshow(img[i,:,:,0], vmin=0, vmax=1)
    ax[i,2].imshow(decoded_img[i,:,:,0], vmin=0, vmax=1)
    ax[i,1].imshow(encoded_img[2][i].reshape(8,4))

ax[0,0].set_title('Greyscale Image')
ax[0,1].set_title('Encoded Image')
ax[0,2].set_title('Reconstructed Image')
plt.show()

