from abc import ABC, abstractmethod

import cv2
import os

from random import seed
from random import randint

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint

from tensorflow.keras.utils import Progbar
from matplotlib import pyplot

from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve

import sys
sys.path.append('../utils')

from models.BaseModel import AbstractModel
from utils.PreProcessing import load_real_samples, generate_samples

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AbstractADDModel(AbstractModel):

    def __init__(self):
        self._reconstruction_error_factor = 0.9
        self._discrimnator_feature_error_factor = 0.1
        self._latent_dimension = 200
        self._generator = None
        self._discriminator = None
        self._feature_extractor = None
        self._anomaly_detector = None
        self._results_dir = None
        self._metrics = None

    def load(self, model_dirs):
        if (self._anomaly_detector is not None):
            self._anomaly_detector.load_weights(model_dirs)
        else:
            raise RuntimeError("Error: The anomaly detector model is None")

    def preprocessing(self, datadir, data_batch_size):
        self._dataset = load_real_samples(datadir, data_batch_size)

    def generate_real_samples(self, n_batch):
        assert self._dataset != None

        return generate_samples(self._dataset, n_batch)

    def train(self, n_epochs, n_batch):
        #asserts
        assert self._feature_extractor != None
        assert self._anomaly_detector != None
        assert self._dataset != None
        assert self._results_dir != None

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
        self._metrics = {
            "loss_hist": loss_hist,
            "rec_loss_hist": rec_loss_hist,
            "disc_loss_hist": disc_loss_hist
        }
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

    def predict(self, img):
        assert self._feature_extractor
        assert self._anomaly_detector

        test_img = None
        if img.ndim < 4:
            test_img = np.asarray([img])
        else:
            test_img = img

        d_x = self._feature_extractor.predict(test_img)
        scores = self._anomaly_detector.evaluate(test_img, [test_img, d_x], verbose=0, steps=1)
        score = scores[-1]
        return score

    def plot(self):
        # asserts
        assert self._anomaly_detector != None
        assert self._feature_extractor != None

        # get a testing image from the dataset
        test_img = self.generate_real_samples(1)[0]
        start = cv2.getTickCount()
        res = self._anomaly_detector.predict(test_img)

        # compute the image reconstrcution score
        score = self.predict(test_img)
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        # print results
        print("Anomaly score : ", score[1])
        print("Processing time: " + str(time))

        # data pos-processing
        test_img2 = (test_img*127.5)+127.5
        test_img2 = test_img2.astype(np.uint8)
        res2 = (res[0]*127.5)+127.5
        res2 = res2.astype(np.uint8)

        # save original image
        pyplot.figure(3, figsize=(3, 3))
        pyplot.title('Original')
        pyplot.imshow(cv2.cvtColor(test_img2[0],cv2.COLOR_BGR2RGB))
        pyplot.savefig(self._results_dir + '/original_sample.pdf')

        # save reconstructed image
        pyplot.figure(3, figsize=(3, 3))
        pyplot.title('Reconstructed')
        pyplot.imshow(cv2.cvtColor(res2[0],cv2.COLOR_BGR2RGB))
        pyplot.savefig(self._results_dir + '/reconstructed_sample.pdf')

    def evaluate_subset(self, test_data, anomaly_treshold):
        # NOTE: this will be the classifier function for the SVC. Rename it to 'classify_subset'
        normal = list()
        anomaly = list()
        progress_bar = Progbar(target=test_data.shape[0])

        # classify the samples between anomaly and healthy
        for i, img in enumerate(test_data):
            # get sample image
            test_img = np.asarray([img])
            score = self.predict(test_img)
            # classify according with the treshold
            if score < anomaly_treshold:
                normal.append(img)
            else:
                anomaly.append(img)
            # display progress
            progress_bar.update(i, values=[('e', score)])

        # convert result arrays to numpy arrays
        normal = np.asarray(normal)
        anomaly = np.asarray(anomaly)
        
        # print results
        print('')
        print('Anomalies:', len(anomaly))
        print('Normal:', len(normal))
        return (normal, anomaly)

    def t_sne_analysis(self, normal, anomaly, output_name='normal_'):
        # asserts
        assert self._feature_extractor != None
        assert self._results_dir != None

        # TODO: remove hardcoded dimensions
        random_image = np.random.uniform(0, 1, (100, 48, 48, 3))

        # intermidieate output of discriminator
        feature_map_of_random = self._feature_extractor.predict(random_image, verbose=1)
        feature_map_of_tomato = self._feature_extractor.predict(anomaly, verbose=1)
        feature_map_of_tomato_1 = self._feature_extractor.predict(normal, verbose=1)

        # t-SNE for visulization
        output = np.concatenate((feature_map_of_random, feature_map_of_tomato, feature_map_of_tomato_1))
        output = output.reshape(output.shape[0], -1)

        # plot t-SNE
        X_embedded = TSNE(n_components=2).fit_transform(output)
        pyplot.figure(5, figsize=(15, 15))
        pyplot.title("t-SNE embedding on the feature representation")
        pyplot.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
        pyplot.scatter(X_embedded[100:(100 + len(anomaly)),0], X_embedded[100:(100 + len(anomaly)),1], label='tomato(anomaly)')
        pyplot.scatter(X_embedded[(100 + len(anomaly)):,0], X_embedded[(100 + len(anomaly)):,1], label='tomato(normal)')
        pyplot.legend()
        pyplot.savefig(self._results_dir + '/' + output_name + 't_sne_results.pdf')

    def evaluate_model(self, normal, anomalies, anomaly_treshold):
        # asserts
        assert self._results_dir != None    

        # initaite variables
        regular_scores = np.zeros(normal.shape[0])
        ano_scores = np.zeros(anomalies.shape[0])
        np_scores = np.zeros(normal.shape[0] + anomalies.shape[0])
        np_testy = np.zeros(normal.shape[0] + anomalies.shape[0])
        progress_bar = Progbar(target=(regular_scores.shape[0] + ano_scores.shape[0]))

        # Eval the regual data
        progress = 0
        for i, img in enumerate(normal):
            score = self.predict(img)
            regular_scores[i] = score
            np_testy[progress] = 0
            np_scores[progress] = score
            progress_bar.update(progress, values=[('e', score)])
            progress = progress + 1
        # analyze subset
        normal_res, anomaly_res = self.evaluate_subset(normal, anomaly_treshold)
        self.t_sne_analysis(normal_res, anomaly_res) 
        pyplot.clf()

        # Eval the anomaly data
        for i, img in enumerate(anomalies):
            score = self.predict(img)
            ano_scores[i] = score
            np_testy[progress] = 1
            np_scores[progress] = score
            progress_bar.update(progress, values=[('e', score)])
            progress = progress + 1
        # analyze subset
        normal_res, anomaly_res = self.evaluate_subset(anomalies, anomaly_treshold)
        self.t_sne_analysis(normal_res, anomaly_res, 'anomaly_')
        pyplot.clf()

        # compute precision and recall curve
        precision, recall, _ = precision_recall_curve(np_testy, np_scores)

        # plot precision and recall
        pyplot.plot(recall, precision, marker='.')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.savefig(self._results_dir + '/precision_recall.pdf')
        
        print('')
        print('Healthy mean:', np.mean(regular_scores))
        print('Healty std:', np.std(regular_scores))
        print('Anomalies mean:', np.mean(ano_scores))
        print('Anomalies std:', np.std(ano_scores))

        # plot scores
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
