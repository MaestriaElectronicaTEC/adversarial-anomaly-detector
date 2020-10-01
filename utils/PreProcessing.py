import cv2
import glob

import numpy as np
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000

#----------------------------------------------------------------------------

def normalize_data(img):
    return (img.astype(np.float32) - 127.5) / 127.5

def load_real_samples(datadir, data_batch_size=64, format='channels_last', input_shape=48):

    trainAug = ImageDataGenerator(
            data_format=format,
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
            target_size=(input_shape, input_shape),
            color_mode="rgb",
            shuffle=True,
            batch_size=data_batch_size)

    return trainGen

def standard_normalization(data, result_dir, scaler_dir=''):
    scaler = None
    if scaler_dir == '':
        scaler = StandardScaler()
        scaler.fit(data)
        dump(scaler, result_dir + "/scaler.joblib") 
    else:
        scaler = load(scaler_dir)

    return scaler.transform(data)

def generate_samples(dataset, n_samples):
    # get batch
    X, _ = dataset.next()
    # choose random instances
    ix = randint(0, X.shape[0], n_samples)
    # select images
    X = X[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def load_test_data(data_dir, dim=48, format='channels_last'):
    img_list = glob.glob(data_dir + '*.png')
    images = list()
    
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_CUBIC)
        img = (img.astype(np.float32) - 127.5) / 127.5

        if format == 'channels_first':
            img = img.transpose([2, 0, 1])

        images.append(img)
        
    print('Found ' + str(len(images)) + ' images for test.')
    return np.asarray(images)

def postprocessing(ref_img, rec_image):
    blur = cv2.blur(ref_img,(8,8))

    cielab_ref = rgb2lab(rec_image)
    cielab_rec = rgb2lab(blur)
    cielab_diff = deltaE_ciede2000(cielab_ref, cielab_rec)

    ret, thresh = cv2.threshold(cielab_diff.astype(np.uint8),127,255,cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    cnts = cv2.findContours(unknown.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    img_copy = ref_img.copy()
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
    img_copy = np.clip(img_copy, 0, 1)
    img_copy = adjust_dynamic_range(img_copy, [0, 1], [0, 255])
    return img_copy.astype(np.uint8)
    