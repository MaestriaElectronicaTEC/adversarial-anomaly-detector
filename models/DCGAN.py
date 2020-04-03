from models.BaseModel import AbstractModel

from os import makedirs

from random import seed
from random import randint

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from keras import backend as K
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
from keras.utils. generic_utils import Progbar
from keras.callbacks import EarlyStopping

from matplotlib import pyplot

import sys
sys.path.append('../utils')

import PreProcessing

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#----------------------------------------------------------------------------

class DCGAN(AbstractModel):

    #----------------------------------------------------------------------------

    # define the standalone generator model
    def define_generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # define model
        model = Sequential()
        # foundation for 6x6 image
        n_nodes = 128 * 6 * 6
        model.add(Dense(n_nodes, kernel_initializer=init, input_dim=self._latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((6, 6, 128)))
        # upsample to 12x12
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 24x24
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 48x48
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # output 48x48x1
        model.add(Conv2D(3, (6,6), activation='tanh', padding='same', kernel_initializer=init))
        return model


    # define the standalone discriminator model
    def define_discriminator(self, in_shape=(48,48,3)): #TODO: Remove hardcoded dimensions
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # define model
        model = Sequential()
        # downsample to 24x24
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # downsample to 12x12
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # downsample to 6x6
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # classifier
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self._generator)
        # add the discriminator
        model.add(self._discriminator)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model


    # select real samples
    def generate_real_samples(self, n_samples):
        # get batch
        X, _ = self._dataset.next()
        # choose random instances
        ix = randint(0, X.shape[0], n_samples)
        # select images
        X = X[ix]
        # generate class labels
        y = ones((n_samples, 1))
        return X, y


    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        x_input = randn(self._latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self._latent_dim)
        return x_input


    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = self._generator.predict(x_input)
        # create class labels
        y = zeros((n_samples, 1))
        return X, y


    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, n_samples=100):
        # prepare fake examples
        X, _ = self.generate_fake_samples(n_samples)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # plot images
        for i in range(10 * 10):
            # define subplot
            pyplot.subplot(10, 10, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(X[i, :, :, 0], cmap=pyplot.cm.gray)
        # save plot to file
        pyplot.savefig(self._results_dir + '/generated_plot_%03d.png' % (step+1))
        pyplot.close()
        # save the generator model
        g_model.save(self._results_dir + '/model_g_%03d.h5' % (step+1))
        d_model.save(self._results_dir + '/model_d_%03d.h5' % (step+1))

    # create a line plot of loss for the gan and save to file
    def plot_history(self, d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(d1_hist, label='d-real')
        pyplot.plot(d2_hist, label='d-fake')
        pyplot.plot(g_hist, label='gen')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(a1_hist, label='acc-real')
        pyplot.plot(a2_hist, label='acc-fake')
        pyplot.legend()
        # save plot to file
        pyplot.savefig(self._results_dir + '/plot_line_plot_loss.png')
        pyplot.close()

    #----------------------------------------------------------------------------

    def __init__(self, latent_dim, results_dir):
        self._latent_dim = latent_dim
        self._generator = self.define_generator()
        self._discriminator = self.define_discriminator()
        self._gan = self.define_gan()
        self._results_dir = results_dir
        # make folder for results
        makedirs(self._results_dir, exist_ok=True)
        super().__init__()

    def load(self, model_dirs):
        self._generator.load_weights(model_dirs['generator'])
        self._discriminator.load_weights(model_dirs['discriminator'])
        self._gan = self.define_gan()

    def preprocessing(self, datadir, data_batch_size=64):
        self._dataset = load_real_samples(datadir, data_batch_size)

    def train(self, n_epochs=10, n_batch=128):
        # calculate the number of samples in half a batch
        half_batch = int(n_batch / 2)
        # prepare lists for storing stats each iteration
        d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
        # manually enumerate epochs
        for epoch in range(n_epochs):
            print ("Epoch:", epoch)
            n_iter = len(self._dataset)
            progress_bar = Progbar(target=n_iter)
            for i in range(n_iter):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # update discriminator model weights
                self._discriminator.trainable = True
                d_loss1, d_acc1 = self._discriminator.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(half_batch)
                # update discriminator model weights
                d_loss2, d_acc2 = self._discriminator.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                self._discriminator.trainable = False
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                progress_bar.update(i, values=[('d1', d_loss1), ('d2', d_loss2), ('g', g_loss), ('a1', d_acc1), ('a2', d_acc2)])
                # record history
                d1_hist.append(d_loss1)
                d2_hist.append(d_loss2)
                g_hist.append(g_loss)
                a1_hist.append(d_acc1)
                a2_hist.append(d_acc2)

            print ('')
            self._dataset.on_epoch_end()
            self.summarize_performance(epoch)

        self.plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

    #----------------------------------------------------------------------------