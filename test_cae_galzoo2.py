from Autoencoder import Models_Cond as Models
import Data.GalZoo2_TFIngest_cond as Data_In

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

tf.keras.backend.clear_session()

layers = 3
filters = [9,5,47]
latent = 20
beta = 10
beta = beta / latent# * x_shape[1]**2 #we need to normalise beta, because it's very dependent on dimensionality
lr = 0.0008317883360924015

test_data = 'Data/Scratch/GalZoo2/galzoo_spiral_flags_test.tfrecords'
batch_size=25000
x_test, c_test = Data_In.create_dataset(test_data, batch_size, None)
steps = 25000//batch_size

with tf.Session() as sess:
    x_test = sess.run(x_test)
    c_test = sess.run(c_test)

encoder2, inputs2, latent_z2 = Models.encoder_model(x_test, c_test, layers, filters, latent, beta)
decoder2 = Models.decoder_model(encoder2, c_test, layers, filters, latent)
vae2 = Models.autoencoder_model(inputs2, decoder2, latent_z2[2])

vae2.load_weights('Data/Scratch/GalZoo2/cae_cnn_beta-10_weights')

decoded_img = vae2.predict([x_test,c_test], batch_size=30, steps=1)
encoded_img = encoder2.predict([x_test,c_test], batch_size=30, steps=1)

f, ax = plt.subplots(3,30)
for i in range(0,30):
    ax[0,i].imshow(x_test[i,:,:,0], vmin=0, vmax=1)
    ax[2,i].imshow(decoded_img[i,:,:,0], vmin=0, vmax=1)
    ax[1,i].imshow(encoded_img[2][i].reshape(5,4))

ax[0,0].set_ylabel('Greyscale Image')
ax[1,0].set_ylabel('Encoded Image')
ax[2,0].set_ylabel('Reconstructed Image')
plt.show()

z = encoded_img[2][20]
z_range = np.linspace(encoded_img[2].min(),encoded_img[2].max(),5)

f, ax = plt.subplots(6,20)
ax[0,4].imshow(x_test[20,:,:,0])
ax[0,5].imshow(decoded_img[20,:,:,0])
for i in range(0,20):
    images = []
    ax[0,i].axis('off')
    for j in range(0,5):
        ax[j+1,i].axis('off')
        z_n = z.copy()
        z_n[i] = z_range[j]
        new_img = decoder2.predict([z_n.reshape(1,20),c_test[20,:].reshape(1,8)], batch_size=1)
        ax[j+1,i].imshow(new_img[0,:,:,0])
        #plt.savefig('Data/Scratch/GalZoo2/imgs/z-'+str(i)+'-step-'+str(j)+'.png')
        #plt.close()
        #images.append(new_img[0,:,:,0])
    #imageio.mimsave('Data/Scratch/GalZoo2/imgs/bayes_z-'+str(i)+'.gif', images)

plt.show()

f, ax = plt.subplots(3,30)

new_cons = np.random.permutation(c_test)

decoded_new_cons = decoder2.predict([encoded_img[2],new_cons], batch_size=30, steps=1)

for i in range(0,30):
    ax[0,i].imshow(x_test[i,:,:,0], vmin=0, vmax=1)
    ax[1,i].imshow(decoded_img[i,:,:,0], vmin=0, vmax=1)
    ax[2,i].imshow(decoded_new_cons[i,:,:,0], vmin=0, vmax=1)

plt.show()

