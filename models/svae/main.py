from __future__ import absolute_import, division, print_function
import math
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import cv2
from sklearn.model_selection import train_test_split
import scipy.misc
from scipy.misc import imsave
from progressbar import ETA, Bar, Percentage, ProgressBar
from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data
from celeba import celeba
from cifar_reader import cifar_reader
import time
from imutils import paths
from ops import *
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1600, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000 , "max epoch")
flags.DEFINE_integer("max_test_epoch", 100, "max  test epoch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_string("working_directory", "/floyd/home/models/svae", "the file directory")
flags.DEFINE_integer("hidden_size", 2, "size of the hidden VAE unit")
flags.DEFINE_integer("channel", 96, "size of initial channel in decoder")
flags.DEFINE_integer("checkpoint", 1450, "number of epochs to be reloaded")
flags.DEFINE_string("model_name", 'low_rank', "vanilla or low_rank")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, or test')
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")
    if FLAGS.model_name == 'vanilla' and FLAGS.hidden_size!= 1:
        sys.exit("the hidden size of vanilla vae must be 1, please check")
        
    ### 0. prepare data

    # seed for reproducibility
    SEED = 44000

    # folder where data is placed
    BASE_FOLDER = '/floyd/input/tomato_dataset/'
    folders = os.listdir(BASE_FOLDER)

    # lists to store data
    num_images = len(list(paths.list_images(BASE_FOLDER)))
    print('Number of images:', num_images)
    data = np.zeros([num_images, 48, 48, 3])
    label = np.ones([num_images]) ## Indicating that is a normal sample

    # loading data to lists
    counter = 0
    for folder in folders:
        for file in os.listdir(BASE_FOLDER + folder + '/'):
            img = cv2.imread(BASE_FOLDER + folder + '/' + file)
            img = cv2.resize(img, (48, 48), interpolation = cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            data[counter,:,:,:] = img
            counter = counter + 1

    # now split the data in to train and test with the help of train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=SEED)
    print ('train shape:', X_train.shape)
    print ('test shape:', X_test.shape)

    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.channel, FLAGS.model_name)
    data = X_train # modify this line to change the dataset reader
    if args.action == 'train':### training part
      for epoch in range(FLAGS.max_epoch): 
          training_loss = 0.0
          pbar = ProgressBar()
          t_start= time.clock()
          for i in pbar(range(FLAGS.updates_per_epoch)):
              images = data[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
              loss_value, kl_loss, rec_loss = model.update_params(images, epoch*FLAGS.updates_per_epoch + i)
              training_loss += loss_value
          t_end = time.clock()
          print ("training per epoch time ====== %f" %(t_end-t_start))
          model.save(epoch)
          training_loss = training_loss/ (FLAGS.updates_per_epoch * FLAGS.batch_size)
          print ("Loss %f" % training_loss)
          model.generate_and_save_images(FLAGS.batch_size, FLAGS.working_directory)
    elif args.action == 'test':  ## evaluation part
        model.reload(FLAGS.checkpoint)
        samples= model.generate_samples()
        sigmas = np.logspace(-1.0, 0.0, 10)
        lls = []
        for sigma in sigmas:
            print("sigma: ", sigma)
            nlls =[]
            for i in range(1, 10+1):
                X= data.next_test_batch(FLAGS.batch_size)
                nll = parzen_cpu_batch(X, samples, sigma=sigma, batch_size=FLAGS.batch_size, num_of_samples=10000, data_size=12288)
                nlls.extend(nll)
            nlls = np.array(nlls).reshape(1000) # 1000 valid images
            print("sigma: ", sigma)
            print("ll: %d" % (np.mean(nlls)))
            lls.append(np.mean(nlls))
        sigma = sigmas[np.argmax(lls)]           

        nlls = []
        data.reset()
        for i in range(1,100+1): # number of test batches = 100
            X= data.next_test_batch(FLAGS.batch_size)
            nll = parzen_cpu_batch(X, samples, sigma=sigma, batch_size=FLAGS.batch_size, num_of_samples=10000, data_size=12288)
            nlls.extend(nll)
        nlls = np.array(nlls).reshape(10000) # 10000 test images
        print("sigma: ", sigma)
        print("ll: %d" % (np.mean(nlls)))
        print("se: %d" % (nlls.std() / np.sqrt(10000)))         
    