from Autoencoder import Models_Cond as Models
import Data.GalZoo2_TFIngest_cond as Data_In

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

"""
Run script for fitting a variational autoencoder to Galaxy Zoo 2 Images

TODO - Fix Multi-gpu implementation, input tensors getting lost when model
       is copied
TODO - standardise input data so we can check GZ2 morphologies
TODO - fix model testing and reconstructions so it picks out same galaxies
TODO - test optimisers
"""

epochs = 1000 #number of epochs to train
batch_size = 200 #batch size (limited by GPU memory)
n_inputs = 10000 #number of objects in x_train
steps_per_epoch = n_inputs//batch_size #tells keras how many bacthes to do
shuffle_size = 100 #how many objects to shuffle (not used right now)


#Load in training and validation data from TFrecords
train_data = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_train.tfrecords' 
valid_data = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_valid.tfrecords'

x_train, c_train = Data_In.create_dataset(train_data, 200, None)
x_valid, c_valid = Data_In.create_dataset(valid_data, 200, None)
"""
with tf.Session() as sess:
    x_train = sess.run(x_train)
    c_train = sess.run(c_train)
    x_valid = sess.run(x_valid)
    c_valid = sess.run(c_valid)

tf.keras.backend.clear_session()

x_train = np.load('Data/Scratch/GalZoo2/galzoo_spiral_train.npy')
c_train = np.load('Data/Scratch/GalZoo2/galzoo_spiral_train_cons.npy')
x_valid = np.load('Data/Scratch/GalZoo2/galzoo_spiral_valid.npy')
c_valid = np.load('Data/Scratch/GalZoo2/galzoo_spiral_valid_cons.npy')
"""
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
filters = [9,5,47]
latent = 20
beta = 10
beta = beta / latent# * x_shape[1]**2 #we need to normalise beta, because it's very dependent on dimensionality
lr = 0.0008317883360924015

#make encoder, pull out pointers to the input layer and latent space
encoder, inputs, latent_z = Models.encoder_model(x_train, c_train, layers, filters, latent, beta)

#make decoder
decoder = Models.decoder_model(encoder, c_train, layers, filters, latent)

#this looks weird, it's calling a function in Models that sets output as decoder(latent_z[2])
vae = Models.autoencoder_model(inputs, decoder, latent_z[2])



#I dunno whats going on here, when you make it a multi-gpu-model it seems to lose
#its connection to x_train, left out for now (it trains pretty quick on 1 V100 tho)

#try:
#    vae = multi_gpu_model(vae, gpus=2)
#    print("Training using multiple GPUs..")
#except ValueError:
#    vae = vae
#    print("Training using single GPU or CPU..")

#compile the model, we make a call to a custom loss function
vae.compile(optimizers.Adam(lr=lr),
            loss = Models.nll,#loss=Models.vae_loss_func(latent_z[0],
            #                   latent_z[1], beta, loss='mse'),
            target_tensors=inputs[0]
           )
vae.add_metric(beta*K.mean(vae.get_layer('encoder').get_layer('kl_divergence_layer').kl_batch), name='kl_loss', aggregation='mean')
vae.add_metric(K.mean(Models.nll(vae.inputs[0], vae.outputs[0])), name='nll_loss', aggregation='mean')

print('vae', vae.losses)

#fit the model, if x_valid is different size to x_train you need to change steps_per_epoch
if tf.is_tensor(x_train)==True:
    history = vae.fit(batch_size=None, epochs=epochs, steps_per_epoch=steps_per_epoch,
                      validation_data=(x_valid, None), validation_steps=steps_per_epoch)
else:
    history = vae.fit(x=[x_train, c_train], y=x_train, batch_size=2000, epochs=epochs,
                      validation_data=([x_valid,c_valid],x_valid))
vae.save_weights('Data/Scratch/GalZoo2/cae_cnn_beta-10_weights') #save the trained weights
#vae.save('Data/Scratch/GalZoo2/cae_cnn_beta-10_model')
#vae.load_model('Data/Scratch/GalZoo2/cae_cnn_beta-10_model.h5', custom_objects={'KLDivergenceLayer' : Models.KLDivergenceLayer})
#vae.load_weights('Data/Scratch/GalZoo2/cae_cnn_beta-10_weights')


fig, ax1 = plt.subplots()

ax1.plot(history.history['loss'], 'k', label='loss')
ax1.plot(history.history['val_loss'], 'k--', label='val_loss')
ax1.plot(history.history['nll_loss'], 'b', label='nll_loss')
ax1.plot(history.history['val_nll_loss'], 'b--', label='val_nll_loss')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(history.history['kl_loss'], 'y', label='kl_loss')
ax2.plot(history.history['val_kl_loss'], 'y--', label='val_kl_loss')
plt.legend()
plt.show()

