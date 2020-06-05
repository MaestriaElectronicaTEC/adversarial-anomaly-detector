from models.BaseModel import AbstractModel

import cv2
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
from tensorflow.keras.utils import Progbar

from matplotlib import pyplot

from sklearn.manifold import TSNE
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
    
    def eval_anomaly(self, img):
        test_img = np.asarray([img])
        d_x = self._feature_extractor.predict(test_img)
        scores = self._anomaly_detector.evaluate(test_img, [test_img, d_x], verbose=0, steps=1)
        score = scores[-1]
        return score

    #----------------------------------------------------------------------------

    def __init__(self, generator, discriminator, results_dir, latent_dimension=200, r_error=0.90, d_error=0.10):
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
            self._dataset.on_epoch_end()
            self._anomaly_detector.save(self._results_dir + '/model_anomaly_%03d.h5' % (epoch+1))

        # save metrics
        metrics = {
            "loss_hist": loss_hist,
            "rec_loss_hist": rec_loss_hist,
            "disc_loss_hist": disc_loss_hist
        }
        self._metrics = metrics
        # plot traning results
        pyplot.plot(loss_hist, label='loss')
        pyplot.plot(rec_loss_hist, label='reconstruction loss')
        pyplot.plot(disc_loss_hist, label='discriminator loss')
        pyplot.xlabel('Epochs')
        # save plot to file
        pyplot.savefig(self._results_dir + '/plot_line_aad_loss.pdf')
        pyplot.close()

    def get_metrics(self):
        return self._metrics

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
        pyplot.savefig(self._results_dir + '/original_sample.pdf')

        pyplot.figure(3, figsize=(3, 3))
        pyplot.title('Reconstructed')
        pyplot.imshow(cv2.cvtColor(res2[0],cv2.COLOR_BGR2RGB))
        pyplot.savefig(self._results_dir + '/reconstructed_sample.pdf')

    def analize_anomalies(self, test_data, anomaly_treshold):
        normal = list()
        anomaly = list()
        progress_bar = Progbar(target=test_data.shape[0])

        for i, img in enumerate(test_data):
            test_img = np.asarray([img])
            #res = anomaly_detector.predict(test_img)
            d_x = self._feature_extractor.predict(test_img)
            scores = self._anomaly_detector.evaluate(test_img, [test_img, d_x], verbose=0, steps=1)

            if scores[-1] < anomaly_treshold:
                normal.append(img)
            else:
                anomaly.append(img)

            progress_bar.update(i, values=[('e', scores[-1])])

        normal = np.asarray(normal)
        anomaly = np.asarray(anomaly)
        
        print('')
        print('Anomalies:', len(anomaly))
        print('Normal:', len(normal))
        return (normal, anomaly)

    def t_sne_analisys(self, normal, anomaly):
        random_image = np.random.uniform(0, 1, (100, 48, 48, 3))

        # intermidieate output of discriminator
        feature_map_of_random = self._feature_extractor.predict(random_image, verbose=1)
        feature_map_of_tomato = self._feature_extractor.predict(anomaly, verbose=1)
        feature_map_of_tomato_1 = self._feature_extractor.predict(normal, verbose=1)

        # t-SNE for visulization
        output = np.concatenate((feature_map_of_random, feature_map_of_tomato, feature_map_of_tomato_1))
        output = output.reshape(output.shape[0], -1)

        X_embedded = TSNE(n_components=2).fit_transform(output)
        pyplot.figure(5, figsize=(15, 15))
        pyplot.title("t-SNE embedding on the feature representation")
        pyplot.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
        pyplot.scatter(X_embedded[100:(100 + len(anomaly)),0], X_embedded[100:(100 + len(anomaly)),1], label='tomato(anomaly)')
        pyplot.scatter(X_embedded[(100 + len(anomaly)):,0], X_embedded[(100 + len(anomaly)):,1], label='tomato(normal)')
        pyplot.legend()
        pyplot.savefig(self._results_dir + '/t_sne_results.pdf')

    def plot_anomalies(self, normal, anomalies):
        regular_scores = np.zeros(normal.shape[0])
        ano_scores = np.zeros(anomalies.shape[0])
        progress_bar = Progbar(target=(regular_scores.shape[0] + ano_scores.shape[0]))

        # Eval the regual data
        progress = 0
        for i, img in enumerate(normal):
            score = self.eval_anomaly(img)
            regular_scores[i] = score
            progress_bar.update(progress, values=[('e', score)])
            progress = progress + 1

        # Eval the anomaly data
        for i, img in enumerate(anomalies):
            score = self.eval_anomaly(img)
            ano_scores[i] = score
            progress_bar.update(progress, values=[('e', score)])
            progress = progress + 1
        
        print('')
        print('Healthy mean:', np.mean(regular_scores))
        print('Healty std:', np.std(regular_scores))
        print('Anomalies mean:', np.mean(ano_scores))
        print('Anomalies std:', np.std(ano_scores))

        reg_mean = np.ones(len(regular_scores)) * np.mean(regular_scores)
        ano_mean = np.ones(len(ano_scores)) * np.mean(ano_scores)

        regular_plot = np.sort(regular_scores)
        ano_plot = np.sort(ano_scores)

        x = np.arange(len(regular_plot))
        y1 = ano_plot
        y2 = regular_plot
        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive'] 

        # Draw Plot 
        fig, ax = pyplot.subplots(1, 1, figsize=(8,4), dpi= 80)
        ax.fill_between(x, y1=y1, y2=0, label='Anomalies', alpha=0.5, color=mycolors[0], linewidth=2)
        ax.fill_between(x, y1=y2, y2=0, label='Healthy samples', alpha=0.5, color=mycolors[1], linewidth=2)

        # Decorations
        ax.legend(loc='best', fontsize=12)
        ax.set(xlim=[0, len(normal)])

        # Draw Tick lines  
        for y in np.arange(2.5, 30.0, 2.5):    
            pyplot.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)

        # Lighten borders
        pyplot.gca().spines["top"].set_alpha(0)
        pyplot.gca().spines["bottom"].set_alpha(.3)
        pyplot.gca().spines["right"].set_alpha(0)
        pyplot.gca().spines["left"].set_alpha(.3)

        pyplot.plot(reg_mean, color=mycolors[1])
        pyplot.plot(ano_mean, color=mycolors[0])
        pyplot.ylabel("Reconstruction score")
        pyplot.xlabel("Samples")

        pyplot.savefig(self._results_dir + '/reconstruction_scores.pdf')

#----------------------------------------------------------------------------
