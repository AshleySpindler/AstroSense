from Autoencoder import Models_TFrecord as Models
import Data.GalZoo2_TFIngest as Data_In
import keras.backend as K
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import optimizers
import matplotlib.pyplot as plt

epochs = 1000
batch_size = 200
steps_per_epoch = 10000//batch_size
shuffle_size = 100

train_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords' 
valid_data = 'Data/Scratch/GalZoo2/galzoo_valid.tfrecords'

x_train = Data_In.create_dataset(train_data, batch_size, shuffle_size)
x_valid = Data_In.create_dataset(valid_data, batch_size, shuffle_size)

layers = 3
filters = [32,32,64]
latent = 32
beta = 1
lr = 1e-4

encoder, inputs, latent_z = Models.encoder_model(x_train, 3, filters, latent)
decoder = Models.decoder_model(encoder, layers, filters, latent)

vae = Models.autoencoder_model(inputs, decoder, latent_z[2])

#try:
#    vae = multi_gpu_model(vae, gpus=2, cpu_relocation=True)
#    print("Training using multiple GPUs..")
#except ValueError:
#    vae = vae
#    print("Training using single GPU or CPU..")

vae.compile(optimizers.Adam(lr=lr), loss=Models.vae_loss_func(latent_z[0], latent_z[1], beta), target_tensors=inputs)

vae.fit(batch_size=None, epochs=epochs, steps_per_epoch=steps_per_epoch,
        validation_data=(x_valid, None), validation_steps=steps_per_epoch)
vae.save_weights('Data/Scratch/GalZoo2/vae_cnn_test.h5')

#plot predictions
test_data = 'Data/Scratch/GalZoo2/galzoo_train.tfrecords'
x_test = Data_In.create_dataset(test_data, batch_size, shuffle_size)
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

