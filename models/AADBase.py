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
from mpl_toolkits.mplot3d import Axes3D

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score, roc_curve, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

import sys
sys.path.append('../utils')

from models.BaseModel import AbstractModel
from utils.PreProcessing import load_real_samples, generate_samples, standard_normalization

import warnings
warnings.filterwarnings("ignore")

from joblib import dump, load

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AbstractADDModel(AbstractModel):

    def __init__(self, format='channels_last', input_shape=48, enable_lpips=False):
        self._reconstruction_error_factor = 0.9
        self._discrimnator_feature_error_factor = 0.1
        self._latent_dimension = 200
        self._generator = None
        self._discriminator = None
        self._feature_extractor = None
        self._anomaly_detector = None
        self._svm = None
        self._results_dir = None
        self._metrics = None
        self._scaler_dir = ''
        self._format = format
        self._input_shape = input_shape
        self._lpips_enabled = enable_lpips

    def load(self, model_dirs):
        if (self._anomaly_detector is not None):
            self._anomaly_detector.load_weights(model_dirs['aad'])
            self._scaler_dir = model_dirs['scaler']

            if (model_dirs['svc'] != ''):
                self._svm = load(model_dirs['svc'])
        else:
            raise RuntimeError("Error: The anomaly detector model is None")

    def preprocessing(self, datadir, data_batch_size=64):
        self._dataset = load_real_samples(datadir, data_batch_size, self._format, self._input_shape)

    def generate_real_samples(self, n_batch):
        assert self._dataset != None

        return generate_samples(self._dataset, n_batch)

    def train(self, n_epochs=10, n_batch=32):
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

                if self._lpips_enabled:
                    loss, rec_loss, disc_loss, lpips_loss = self._anomaly_detector.train_on_batch(x, [x, d_x, x])

                    # summarize loss on this batch
                    progress_bar.update(i, values=[('loss', loss), ('rec loss', rec_loss), ('disc loss', disc_loss), ('lpips loss', lpips_loss)])
                else:
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

    def train_svm_with_grid_search(self, normal, anomaly):
        # estimate normal and anomalies data
        _, _, np_testy, np_scores = self.estimate_labeled_data(normal, anomaly)

        # hyperparameters
        param_grid = [
            {'C': [1, 100, 1000, 10000], 'kernel': ['linear']},
            {'C': [1, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001], 'degree': [2, 3], 'kernel': ['poly','rbf','sigmoid']},
        ]

        # support vector machine
        self._svm = svm.SVC(verbose=False)
        clf = GridSearchCV(self._svm, param_grid, verbose=True, cv=10, n_jobs=4)

        # data preprocessing
        #data = np_scores.reshape(-1, 1)
        data = standard_normalization(np_scores, self._results_dir)
        #X_train, _, y_train, _ = train_test_split(data, np_testy, test_size=0, random_state=42)
        X_train = data
        y_train = np_testy

        print("")
        print("X_train:", len(X_train))
        print("Y_train:", len(y_train))
        print("")
        print("Training...")

        # train the model
        clf.fit(X_train, y_train)

        print("")
        print("Results:")

        print("")
        print("Best estimator")
        print(clf.best_estimator_)

        print("")
        print("Best parameters")
        print(clf.best_params_)

        print("")
        print("Best score:", clf.best_score_)

    def train_svm(self, C, gamma, degree, kernel, normal, anomaly):
        # estimate normal and anomalies data
        _, _, np_testy, np_scores = self.estimate_labeled_data(normal, anomaly)

        # support vector machine
        if (gamma is -1):
            gamma = 'scale'
        self._svm = svm.SVC(C=C, gamma=gamma, degree=degree, kernel=kernel, probability=True, verbose=True)
        print("")
        print(self._svm)

        # data preprocessing
        data = standard_normalization(np_scores, self._results_dir)
        X_train, X_test, y_train, y_test = train_test_split(data, np_testy, test_size=0.3, random_state=20)

        print("")
        print("X_train:", len(X_train))
        print("Y_train:", len(y_train))
        print("")
        print("Training...")

        # train the model
        self._svm.fit(X_train, y_train)

        # save the SVC model
        dump(self._svm, self._results_dir + "/svc.joblib") 

        # plot resutls

        # Precision and recall
        y_score = self._svm.decision_function(X_test)
        average_precision = average_precision_score(y_test, y_score)
        disp = plot_precision_recall_curve(self._svm, X_test, y_test)
        #disp.ax_.set_title('2-class Precision-Recall curve: '
        #                   'AP={0:0.2f}'.format(average_precision))
        pyplot.savefig(self._results_dir + '/precision_and_recall.pdf')
        pyplot.show()

        # ROC
        # predict probabilities
        yhat = self._svm.predict_proba(X_test)
        # retrieve just the probabilities for the positive class
        pos_probs = yhat[:, 1]
        # plot no skill roc curve
        pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        AUC = auc(fpr, tpr)
        # plot model roc curve
        pyplot.plot(fpr, tpr, label='SVC')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # save figure
        pyplot.savefig(self._results_dir + '/roc.pdf')
        # show the plot
        pyplot.show()

        # show SVC scattered graph
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, s=50, cmap='Paired')
        pyplot.show()

        # get confusion matrix
        y_pred = self._svm.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        presicion  = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("average presicion: " + str(average_precision))
        print("presicion: " + str(presicion))
        print("recall: " + str(recall))
        print("sensitivity: " + str(sensitivity))
        print("specificity: " + str(specificity))
        print("AUC: " + str(AUC))

    def get_metrics(self):
        return self._metrics

    def predict(self, img):
        assert self._feature_extractor
        assert self._anomaly_detector

        # prepare input data
        test_img = None
        if img.ndim < 4:
            test_img = np.asarray([img])
        else:
            test_img = img

        # get loss values
        d_x = self._feature_extractor.predict(test_img)
        scores = self._anomaly_detector.evaluate(test_img, [test_img, d_x], verbose=0, steps=1)
        np_scores = np.zeros([1, 3])
        np_scores[0, 0] = scores[-1]
        np_scores[0, 1] = scores[0]
        np_scores[0, 2] = scores[1]
        data = standard_normalization(np_scores, self._results_dir, self._scaler_dir)
        
        # get reconstrcuted 
        res = self._anomaly_detector.predict(test_img)

        # data pos-processing
        res2 = np.clip((res[0]+1)*0.5, 0, 1)

        if (self._svm is not None):
            # classify
            class_predicted = self._svm.predict(data)

            return np_scores, class_predicted[0], res2
        else:
            return np_scores, -1, res2

    def plot(self):
        # asserts
        assert self._anomaly_detector != None
        assert self._feature_extractor != None

        # get a testing image from the dataset
        test_img = self.generate_real_samples(1)[0]
        start = cv2.getTickCount()

        score, class_predicted, res = self.predict(test_img)

        # compute the image reconstrcution score
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        # classify image
        if (class_predicted == 0):
            print("Image is a healthy sample")
        else:
            print("Image has anomalies!")

        # print results
        print("Anomaly score : ", score[0, 0])
        print("Processing time: " + str(time))

        # data pos-processing
        test_img2 = (test_img*127.5)+127.5
        test_img2 = test_img2.astype(np.uint8)

        # save original image
        pyplot.figure(3, figsize=(3, 3))
        #pyplot.title('Original')
        pyplot.axis('off')
        pyplot.imshow(test_img2[0])
        pyplot.savefig(self._results_dir + '/original_sample.pdf')

        pyplot.show()

        # save reconstructed image
        pyplot.figure(3, figsize=(3, 3))
        #pyplot.title('Reconstructed')
        pyplot.axis('off')
        pyplot.imshow(res[0])
        pyplot.savefig(self._results_dir + '/reconstructed_sample.pdf')

        pyplot.show()

    def evaluate_subset(self, test_data):
        normal = list()
        anomaly = list()
        progress_bar = Progbar(target=test_data.shape[0])

        # classify the samples between anomaly and healthy
        for i, img in enumerate(test_data):
            # get sample image
            test_img = np.asarray([img])
            score, class_predicted, _ = self.predict(test_img)
            # classify according with the treshold
            if class_predicted == 0:
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
        #pyplot.figure(5, figsize=(15, 15))
        #pyplot.title("t-SNE embedding on the feature representation")
        #pyplot.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
        pyplot.scatter(X_embedded[100:(100 + len(anomaly)),0], X_embedded[100:(100 + len(anomaly)),1], label='tomato(anomaly)')
        pyplot.scatter(X_embedded[(100 + len(anomaly)):,0], X_embedded[(100 + len(anomaly)):,1], label='tomato(normal)')
        pyplot.legend()
        pyplot.savefig(self._results_dir + '/' + output_name + 't_sne_results.pdf')

    def estimate_labeled_data(self, normal, anomalies):
        # initaite variables
        regular_scores = np.zeros(normal.shape[0])
        ano_scores = np.zeros(anomalies.shape[0])
        np_scores = np.zeros([normal.shape[0] + anomalies.shape[0], 3])
        np_testy = np.zeros(normal.shape[0] + anomalies.shape[0])
        progress_bar = Progbar(target=(regular_scores.shape[0] + ano_scores.shape[0]))

        # Eval the regual data
        progress = 0
        for i, img in enumerate(normal):
            score, _, _ = self.predict(img)
            regular_scores[i] = score[0, 0]
            np_testy[progress] = 0
            np_scores[progress] = score[0]
            progress_bar.update(progress, values=[('e', score[0, 0])])
            progress = progress + 1

        # Eval the anomaly data
        for i, img in enumerate(anomalies):
            score, _, _ = self.predict(img)
            ano_scores[i] = score[0, 0]
            np_testy[progress] = 1
            np_scores[progress] = score[0]
            progress_bar.update(progress, values=[('e', score[0, 0])])
            progress = progress + 1

        return (regular_scores, ano_scores, np_testy, np_scores)
    
    def evaluate_model(self, normal, anomalies):
        # asserts
        assert self._results_dir != None

        # estimate normal and anomalies data
        regular_scores, ano_scores, _, _ = self.estimate_labeled_data(normal, anomalies) 

        # t-SNE analysis
        #self.t_sne_analysis()
        
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
