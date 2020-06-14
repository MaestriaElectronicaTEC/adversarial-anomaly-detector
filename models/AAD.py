from models.AADBase import AbstractADDModel

import cv2
import os
from os import makedirs

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

#----------------------------------------------------------------------------

class AAD(AbstractADDModel):

    #----------------------------------------------------------------------------

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

    def __init__(self, generator, discriminator, results_dir, latent_dimension=200, r_error=0.90, d_error=0.10):
        super().__init__()
        self._reconstruction_error_factor = r_error
        self._discrimnator_feature_error_factor = d_error
        self._latent_dimension = latent_dimension
        self._generator = generator
        self._discriminator = discriminator
        self._feature_extractor = self.define_feature_extractor()
        self._anomaly_detector = self.define_anomaly_detector()
        self._results_dir = results_dir
        self._metrics = None
        # make folder for results
        makedirs(self._results_dir, exist_ok=True)


#----------------------------------------------------------------------------
