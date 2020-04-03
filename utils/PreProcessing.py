import numpy as np

from keras.preprocessing.image import ImageDataGenerator

#----------------------------------------------------------------------------

def normalize_data(img):
    return (img.astype(np.float32) - 127.5) / 127.5

def load_real_samples(datadir, data_batch_size=64):

    trainAug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
            preprocessing_function=normalize_data)

    trainGen = trainAug.flow_from_directory(
            datadir,
            class_mode="input",
            target_size=(48, 48), #TODO: Remove hardcoded dimensions
            color_mode="rgb",
            shuffle=True,
            batch_size=data_batch_size)

    return trainGen
