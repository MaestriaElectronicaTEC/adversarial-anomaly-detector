from models.AADBase import AbstractADDModel

import cv2
import os
from os import makedirs
import math

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
from tensorflow.keras.layers import LocallyConnected1D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

DROPOUT = 0.2
TOP_DOWN_PYRAMID_SIZE = 256

#----------------------------------------------------------------------------

class StyleAAD2(AbstractADDModel):

    #----------------------------------------------------------------------------

    def is_square(self, n):
        return (n == int(math.sqrt(n) + 0.5)**2)

    def grouped_convolution_block(self, input, grouped_channels, cardinality, strides, weight_decay=5e-4):
        init = input
        group_list = []

        if cardinality == 1:
            # with cardinality 1, it is a standard convolution
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                    kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            x = BatchNormalization(axis=3)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(DROPOUT)(x)
            return x

        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                    kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
            group_list.append(x)

        group_merge = Concatenate(axis=3)(group_list)
        x = BatchNormalization(axis=3)(group_merge)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(DROPOUT)(x)
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
        x = Dropout(DROPOUT)(x)

        x = self.grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
        x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=3)(x)

        x = Add()([init, x])
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(DROPOUT)(x)
        return x

    def map2style(self, x, model_scale, depth):
        layer_size = model_scale*8*8*8
        if self.is_square(layer_size): # work out layer dimensions
            layer_l = int(math.sqrt(layer_size)+0.5)
            layer_r = layer_l
        else:
            layer_m = math.log(math.sqrt(layer_size),2)
            layer_l = 2**math.ceil(layer_m)
            layer_r = layer_size // layer_l
        layer_l = int(layer_l)
        layer_r = int(layer_r)

        x_init = None

        if (depth < 0):
            depth = 1

        x = Reshape((256, 256))(x) # all weights used

        while (depth > 0): # See https://github.com/OliverRichter/TreeConnect/blob/master/cifar.py - TreeConnect inspired layers instead of dense layers.
            x = LocallyConnected1D(layer_r, 1)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(DROPOUT)(x)
            x = Permute((2, 1))(x)
            x = LocallyConnected1D(layer_l, 1)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(DROPOUT)(x)
            x = Permute((2, 1))(x)
            if x_init is not None:
                x = Add()([x, x_init])   # add skip connection
            x_init = x
            depth-=1

        x = Reshape((model_scale, 512))(x) # train against all dlatent values
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
    def define_anomaly_detector(self, cardinality=32, width=4, weight_decay=5e-4, batch_norm=True, batch_momentum=0.9):
        self._feature_extractor.trainable = False
        self._generator.trainable = False
        self._generator.model_mapping.trainable = False
        self._generator.model_synthesis.trainable = False

        # encoder
        size = 2
        depth = 2
        depths = ()
        bn_axis = 3
        model_res = 64
        model_scale = int(2*(math.log(model_res,2)-1))

        if size == 1:
            depths = (3, 4, 6, 3)
        if size == 2:
            depths = (3, 4, 23, 3)
        if size >= 3:
            depths = (3, 8, 23, 3)

        input_tensor = Input(shape=(3,64,64))
        resnext = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)

        if batch_norm:
            resnext = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(resnext)
        resnext = LeakyReLU(alpha=0.2)(resnext)
        resnext = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(resnext)
        resnext = Dropout(DROPOUT)(resnext)
        stage_1 = resnext

        # filters are cardinality * width * 2 for each depth level
        for _ in range(depths[0]):
            resnext = self.bottleneck_block(resnext, 128, cardinality, strides=1, weight_decay=weight_decay)
        stage_2 = resnext

        resnext = self.bottleneck_block(resnext, 256, cardinality, strides=2, weight_decay=weight_decay)
        for _ in range(1, depths[1]):
            resnext = self.bottleneck_block(resnext, 256, cardinality, strides=1, weight_decay=weight_decay)
        stage_3 = resnext

        P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(stage_3)
        P2 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                    Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
        
        # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
        P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p2")(P2)
        P2 = LeakyReLU(alpha=0.2)(P2)
        P2 = Dropout(DROPOUT)(P2)
        P2 = Conv2D(2048, 1)(P2)
        P2 = LeakyReLU(alpha=0.2)(P2)
        P2 = Dropout(DROPOUT)(P2)

        P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="same", name="fpn_p3")(P3)
        P3 = LeakyReLU(alpha=0.2)(P3)
        P3 = Dropout(DROPOUT)(P3)
        P3 = Conv2D(8192, 1)(P3)
        P3 = LeakyReLU(alpha=0.2)(P3)
        P3 = Dropout(DROPOUT)(P3)

        output_1 = self.map2style(P2, int(model_scale/2), depth)
        output_2 = self.map2style(P3, int(model_scale/2), depth)

        x = Concatenate(axis=1)([output_1, output_2])

        # G & D feature
        G_out = self._generator.model_synthesis(x)
        D_out= self._feature_extractor(G_out)
        model = Model(inputs=input_tensor, outputs=[G_out, D_out])
        model.compile(loss=self.sum_of_residual, loss_weights= [self._reconstruction_error_factor, self._discrimnator_feature_error_factor], optimizer='adam')

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
