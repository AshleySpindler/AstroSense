import numpy as np
import keras as k
import matplotlib.pyplot as plt
import os

from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.utils import plot_model
from keras.datasets import mnist

#TODO build VAE class
class VAE:

    def __init__(self, input_shape, layers, filters, latent_dims):
        self.encoder = self.encoder_model(input_shape, layers, filters, latent_dims)
        self.decoder = self.decoder_model(input_shape, layers, filters, latent_dims)
        self.inputs = self.encoder.inputs
        self.outputs = self.decoder(self.encoder(self.encoder.inputs)[2])
        self.autoencoder = Model(self.inputs, self.outputs, name='Autoencoder')

    #TODO build encoder model, variable n. of layers, mean and sigma latent space
    def encoder_model(self, input_shape, layers, filters, latent_dims):
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
        l.append(Input(shape=input_shape, name='encoder_input'))

        for i in range(0,layers):
            l.append(Conv2D(filters[i], (5,5), padding='same',
                            data_format='channels_last', name='Conv_'+str(i),
                            activation='relu')(l[i*2]))
            l.append(MaxPooling2D(pool_size=(2, 2),
                     data_format='channels_last',
                     name='Pool_'+str(i))(l[i*2+1]))

        conv_out = Flatten(name='Flatten_dims')(l[-1])

        self.z_mean = Dense(latent_dims, name='z_mean')(conv_out)
        self.z_sigma = Dense(latent_dims, name='z_sigma')(conv_out)
        self.z = Lambda(VAE_sampling, output_shape=(latent_dims,), name='z')([self.z_mean,self.z_sigma])
        encoder = Model(l[0], [self.z_mean,self.z_sigma,self.z], name='encoder')

        return encoder

    def decoder_model(self, input_shape, layers, filters, latent_dims):
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

        flat_dims = self.encoder.get_layer('Flatten_dims').output_shape
        pool_dims = self.encoder.get_layer('Flatten_dims').input_shape

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

        decoder = Model(latent_inputs, l[-1], name='decoder')

        return decoder

    def compile_model(self):

        self.reconstruction_loss = mse(K.flatten(self.inputs), 
                                       K.flatten(self.outputs))
        self.reconstruction_loss *= self.autoencoder.input_shape[1]*self.autoencoder.input_shape[1]

        self.kl_loss = (1 + self.z_sigma - 
                  K.square(self.z_mean) - 
                  K.exp(self.z_sigma))
        self.kl_loss = K.sum(self.kl_loss, axis=-1)
        self.kl_loss *= -0.5

        vae_loss = K.mean(self.reconstruction_loss+self.kl_loss)

        self.autoencoder.add_loss(vae_loss)

        self.autoencoder.compile(optimizer='adam')

    def plot_models(self):
        self.autoencoder.summary()
        self.encoder.summary()
        self.decoder.summary()
        plot_model(self.autoencoder, to_file='vae_autoencoder.png', show_shapes=True)
        plot_model(self.encoder, to_file='vae_encoder.png', show_shapes=True)
        plot_model(self.decoder, to_file='vae_decoder.png', show_shapes=True)


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

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = x_train.shape[1:]
    layers = 2
    filters = [1,5]
    latent = 2
    epochs = 50
    batch_size = 128
    vae = VAE(input_shape, layers, filters, latent)
    vae.compile_model()
    #vae.autoencoder.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
    #vae.autoencoder.save_weights('vae_mlp_mnist.h5')
    vae.autoencoder.load_weights('vae_mlp_mnist.h5')
    models = (vae.encoder, vae.decoder)
    data = (x_test, y_test)
    plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")


