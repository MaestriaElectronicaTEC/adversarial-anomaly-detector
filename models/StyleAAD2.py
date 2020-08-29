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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#----------------------------------------------------------------------------

class StyleAAD2(AbstractADDModel):

    #----------------------------------------------------------------------------

    def grouped_convolution_block(self, input, grouped_channels, cardinality, strides, weight_decay=5e-4):
        init = input
        group_list = []

        if cardinality == 1:
            # with cardinality 1, it is a standard convolution
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                    kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            x = BatchNormalization(axis=3)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.05)(x)
            return x

        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                    kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
            group_list.append(x)

        group_merge = Concatenate(axis=3)(group_list)
        x = BatchNormalization(axis=3)(group_merge)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)
        return x

    def bottleneck_block(self, input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
        init = input
        grouped_channels = int(filters / cardinality)

        if init.shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=3)(init)

        x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
                kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)

        x = self.grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
        x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([init, x])
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)
        return x

    # anomaly loss function
    def sum_of_residual(self, y_true, y_pred):
        return K.sum(K.abs(y_true - y_pred))

    # discriminator intermediate layer feautre extraction
    def define_feature_extractor(self):
        # define model
        intermidiate_model = Model(inputs=self._discriminator.model.layers[0].input, outputs=self._discriminator.model.layers[27].output)
        intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return intermidiate_model

    # anomaly detection model define
    def define_anomaly_detector(self, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True, batch_momentum=0.9):
        self._feature_extractor.trainable = False
        self._generator.trainable = False
        self._generator.model_mapping.trainable = False
        self._generator.model_synthesis.trainable = False

        # define custom activation function
        def custom_tanh(x):
            return 5*tf.math.tanh(x)

        # encoder
        bn_axis = 3

        input_tensor = Input(shape=(3,64,64))
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)

        if batch_norm:
            x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = Dropout(0.05)(x)

        # filters are cardinality * width * 2 for each depth level
        for _ in range(depth[0]):
            x = self.bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)

        x = self.bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay)
        for _ in range(1, depth[1]):
            x = self.bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)

        x = self.bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)
        for _ in range(1, depth[2]):
            x = self.bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)

        x = Flatten()(x)
        x = Dense(self._latent_dimension, trainable=True)(x)
        x = Activation(custom_tanh)(x)
        encoder = Model(inputs=input_tensor, outputs=x)

        # G & D feature
        G_mapping_out = self._generator.model_mapping(encoder.output)
        G_out = self._generator.model_synthesis(G_mapping_out)
        D_out= self._feature_extractor(G_out)
        model = Model(inputs=input_tensor, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights= [self._reconstruction_error_factor, self._discrimnator_feature_error_factor], optimizer='adam')

        # batchnorm learning phase fixed (test) : make non trainable
        K.set_learning_phase(0)
        return model

    #----------------------------------------------------------------------------

    def __init__(self, generator, discriminator, results_dir, latent_dimension=200, r_error=0.90, d_error=0.10):
        super().__init__(format='channels_first', input_shape=64)
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
