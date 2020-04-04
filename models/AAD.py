from models.BaseModel import AbstractModel

#import cv2
import os
from os import makedirs

from random import seed
from random import randint

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from tensorflow.keras import backend as K
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot

import sys
sys.path.append('../utils')

from utils.PreProcessing import load_real_samples

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#----------------------------------------------------------------------------

class AAD(AbstractModel):

    #----------------------------------------------------------------------------

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

    # anomaly loss function
    def sum_of_residual(self, y_true, y_pred):
        return K.sum(K.abs(y_true - y_pred))

    # discriminator intermediate layer feautre extraction
    def define_feature_extractor(self, in_shape=(48,48,3)): #TODO: Remove hardcoded dimensions
        # define model
        intermidiate_model = Sequential()
        # downsample to 24x24
        intermidiate_model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape, weights=self._discriminator.layers[0].get_weights()))
        intermidiate_model.add(BatchNormalization())
        intermidiate_model.add(LeakyReLU(alpha=0.2))
        intermidiate_model.add(Dropout(0.25))
        # downsample to 12x12
        intermidiate_model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', weights=self._discriminator.layers[4].get_weights()))
        intermidiate_model.add(BatchNormalization())
        intermidiate_model.add(LeakyReLU(alpha=0.2))
        intermidiate_model.add(Dropout(0.25))
        # downsample to 6x6
        intermidiate_model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', weights=self._discriminator.layers[8].get_weights()))
        intermidiate_model.add(BatchNormalization())
        intermidiate_model.add(LeakyReLU(alpha=0.2))
        intermidiate_model.add(Dropout(0.25))
        intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
        return intermidiate_model

    # anomaly detection model define
    def define_anomaly_detector(self):
        self._feature_extractor.trainable = False
        self._generator.trainable = False
        # Encoder
        aInput = Input(shape=(48,48,3)) #TODO: Remove hardcoded dimensions
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', trainable=True)(aInput)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        # downsample to 12x12
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', trainable=True)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        # downsample to 6x6
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', trainable=True)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        # laten space
        x = Flatten()(x)
        encoder = Dense(self._latent_dimension, activation='sigmoid', trainable=True)(x)
        # G & D feature
        G_out = self._generator(encoder)
        D_out= self._feature_extractor(G_out)
        model = Model(inputs=aInput, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights= [self._reconstruction_error_factor, self._discrimnator_feature_error_factor], optimizer='adam')
        # batchnorm learning phase fixed (test) : make non trainable
        K.set_learning_phase(0)
        return model

    #----------------------------------------------------------------------------

    def __init__(self, generator, discriminator, results_dir, latent_dimension=200):
        self._reconstruction_error_factor = 0.90
        self._discrimnator_feature_error_factor = 0.10
        self._latent_dimension = latent_dimension
        self._generator = generator
        self._discriminator = discriminator
        self._feature_extractor = self.define_feature_extractor()
        self._anomaly_detector = self.define_anomaly_detector()
        self._results_dir = results_dir
        # make folder for results
        makedirs(self._results_dir, exist_ok=True)
        super().__init__()

    def load(self, model_dirs):
        self._anomaly_detector.load_weights(model_dirs)

    def preprocessing(self, datadir, data_batch_size=64):
        self._dataset = load_real_samples(datadir, data_batch_size)

    def train(self, n_epochs=10, n_batch=128):
        # prepare lists for storing stats each iteration
        loss_hist, rec_loss_hist, disc_loss_hist = list(), list(), list()
        loss, rec_loss, disc_loss = 0, 0, 0
        # manually enumerate epochs
        for epoch in range(n_epochs):
            print ("Epoch:", epoch)
            n_iter = len(self._dataset)
            progress_bar = Progbar(target=n_iter)
            for i in range(n_iter):
                # get randomly selected 'real' samples
                x, _ = self.generate_real_samples(n_batch)

                d_x = self._feature_extractor.predict(x)
                loss, rec_loss, disc_loss = self._anomaly_detector.train_on_batch(x, [x, d_x])

                # summarize loss on this batch
                progress_bar.update(i, values=[('loss', loss), ('rec loss', rec_loss), ('disc loss', disc_loss)])

            # record history
            loss_hist.append(loss)
            rec_loss_hist.append(rec_loss)
            disc_loss_hist.append(disc_loss)

            print ('')
            dataset.on_epoch_end()
            self._anomaly_detector.save(self._results_dir + '/model_anomaly_%03d.h5' % (epoch+1))

    def plot(self):
        test_img = self.generate_real_samples(1)[0]
        start = cv2.getTickCount()
        res = self._anomaly_detector.predict(test_img)

        d_x = self._feature_extractor.predict(test_img)
        score = self._anomaly_detector.evaluate(test_img, [test_img, d_x], verbose=0, steps=1)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        print("anomaly score : ", score[1])

        test_img2 = (test_img*127.5)+127.5
        test_img2 = test_img2.astype(np.uint8)
        res2 = (res[0]*127.5)+127.5
        res2 = res2.astype(np.uint8)

        pyplot.figure(3, figsize=(3, 3))
        pyplot.title('Original')
        pyplot.imshow(cv2.cvtColor(test_img2[0],cv2.COLOR_BGR2RGB))
        pyplot.savefig(self._results_dir + '/original_sample.png')

        pyplot.figure(3, figsize=(3, 3))
        pyplot.title('Reconstructed')
        pyplot.imshow(cv2.cvtColor(res2[0],cv2.COLOR_BGR2RGB))
        pyplot.savefig(self._results_dir + '/reconstructed_sample.png')

#----------------------------------------------------------------------------
