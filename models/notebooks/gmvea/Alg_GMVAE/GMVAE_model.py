#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:24:14 2018

@author: psanch
"""
from base.base_model import BaseModel
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.utils. generic_utils import Progbar
import numpy as np
from GMVAE_graph import GMVAEGraph
from GMVAECNN_graph import GMVAECNNGraph

from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import sys

import utils.utils as utils
import utils.constants as const

from numpy.random import randn
from numpy.random import randint

import cv2

class GMVAEModel(BaseModel):
    def __init__(self,network_params,sigma=0.001, sigma_act=tf.nn.softplus,
                 transfer_fct= tf.nn.relu,learning_rate=0.002,
                 kinit=tf.contrib.layers.xavier_initializer(),batch_size=32,
                 drop_rate=0., epochs=200, checkpoint_dir='', 
                 summary_dir='', result_dir='', restore=0, model_type=0):
        super().__init__(checkpoint_dir, summary_dir, result_dir)
        
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.z_file = result_dir + '/z_file'
    
        self.restore = restore
        
        
        # Creating computational graph for train and test
        self.graph = tf.Graph()
        with self.graph.as_default():
            if(model_type == const.GMVAE):
                self.model_graph = GMVAEGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)
            if(model_type == const.GMVAECNN):
                self.model_graph = GMVAECNNGraph(network_params,sigma, sigma_act,
                                          transfer_fct,learning_rate, kinit,batch_size,
                                          reuse=False)          

            self.model_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            # model_vars = tf.trainable_variables()
            # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            
    
    def train_epoch(self, session,logger, data_train, beta=1):
        loop = len(data_train)#tqdm(range(data_train.num_batches(self.batch_size)))
        #progress_bar = Progbar(target=loop)
        losses = []
        recons = []
        cond_prior = []
        KL_w = []
        y_prior = []
        L2_loss = []
        
        for i in range(loop):
            # get batch
            X, _ = data_train.next()
            # choose random instances
            ix = randint(0, X.shape[0], self.batch_size)
            # select images
            batch_x = X[ix]
            
            #batch_x = next(data_train.next_batch(self.batch_size))
            # loss_aux, recon_aux, cond_prior_aux, KL_w_aux, y_prior_aux, L2_loss_aux
            loss_list = self.model_graph.partial_fit(session, batch_x, beta, self.drop_rate)
            #progress_bar.update(i, values=[('loss', loss_list[0]), ('recons', loss_list[1])])
            losses.append(loss_list[0])
            recons.append(loss_list[1])
            cond_prior.append(loss_list[2])
            KL_w.append(loss_list[3])
            y_prior.append(loss_list[4])
            L2_loss.append(loss_list[5])
        
        losses = np.mean(losses)
        recons = np.mean(recons)
        cond_prior = np.mean(cond_prior)
        KL_w = np.mean(KL_w)
        y_prior = np.mean(y_prior)
        L2_loss = np.mean(L2_loss)      

        
        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': losses,
            'recons_loss': recons,
            'CP_loss': cond_prior,
            'KL_w_loss': KL_w,
            'y_p_loss': y_prior,
            'L2_loss': L2_loss
        }
        
        logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        return losses, recons, cond_prior, KL_w, y_prior, L2_loss
        
    def valid_epoch(self, session, logger, data_valid,beta=1):
        # COMPUTE VALID LOSS
        #loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        loop = len(data_valid)
        losses = []
        recons = []
        cond_prior = []
        KL_w = []
        y_prior = []
        L2_loss = []

        for _ in range(loop):
            # get batch
            X, _ = data_valid.next()
            # choose random instances
            ix = randint(0, X.shape[0], self.batch_size)
            # select images
            batch_x = X[ix]
            
            #batch_x = next(data_valid.next_batch(self.batch_size))
            loss_list = self.model_graph.evaluate(session, batch_x, beta)
            
            losses.append(loss_list[0])
            recons.append(loss_list[1])
            cond_prior.append(loss_list[2])
            KL_w.append(loss_list[3])
            y_prior.append(loss_list[4])
            L2_loss.append(loss_list[5])

        losses = np.mean(losses)
        recons = np.mean(recons)
        cond_prior = np.mean(cond_prior)
        KL_w = np.mean(KL_w)
        y_prior = np.mean(y_prior)
        L2_loss = np.mean(L2_loss)      

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'loss': losses,
            'recons_loss': recons,
            'CP_loss': cond_prior,
            'KL_w_loss': KL_w,
            'y_p_loss': y_prior,
            'L2_loss': L2_loss
        }
        logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict)
        
        return losses, recons, cond_prior, KL_w, y_prior, L2_loss
        
    def train(self, data_train, data_valid, enable_es=1):
        
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)
            
            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            early_stopping = EarlyStopping()
            
            if(self.restore==1 and self.load(session, saver) ):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)      
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
                
                   
            if(self.model_graph.cur_epoch_tensor.eval(session) ==  self.epochs):
                return
            
            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):
        
                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch
                # beta=utils.sigmoid(cur_epoch- 50)
                beta = 1.
                losses, recons, cond_prior, KL_w, y_prior, L2_loss = self.train_epoch(session, logger, data_train, beta=beta)
                train_string = 'TRAIN | Loss: ' + str(losses) + \
                            ' | Recons: ' + str(recons) + \
                            ' | CP: ' + str(cond_prior) + \
                            ' | KL_w: ' + str(KL_w) + \
                            ' | KL_y: ' + str(y_prior) + \
                            ' | L2_loss: '+  str(L2_loss)
                            
                if np.isnan(losses):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    print('Recons: ', recons)
                    print('CP: ', cond_prior)
                    print('KL_w: ', KL_w)
                    print('KL_y: ', y_prior)
                    sys.exit()
                    
                loss_val, recons, cond_prior, KL_w, y_prior, L2_loss = self.valid_epoch(session, logger, data_valid, beta=beta)
                valid_string = 'VALID | Loss: ' + str(loss_val) + \
                            ' | Recons: ' + str(recons) + \
                            ' | CP: ' + str(cond_prior) + \
                            ' | KL_w: ' + str(KL_w) + \
                            ' | KL_y: ' + str(y_prior) + \
                            ' | L2_loss: '+  str(L2_loss)
                
                print('')
                print(train_string)
                print(valid_string)
                
                if(cur_epoch>0 and cur_epoch % 10 == 0):
                    self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
                    
                session.run(self.model_graph.increment_cur_epoch_tensor)
                
                #Early stopping
                if(enable_es==1 and early_stopping.stop(loss_val)):
                    print('Early Stopping!')
                    break
                    
        
                self.save(session,saver, self.model_graph.global_step_tensor.eval(session))
                data_train.on_epoch_end()
                data_valid.on_epoch_end()

        return
    
    def generate_samples(self, data,num_batches=20):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            #x_batch = data.random_batch(self.batch_size)
            # choose random instances
            ix = randint(0, data.shape[0], self.batch_size)
            # select images
            x_batch = data[ix]
            x_samples,  z_samples,  w_samples = self.model_graph.generate_samples(session, x_batch, beta=1, num_batches=num_batches)
            
            return x_samples,  z_samples, w_samples
    
    def eval_anomaly(self, test_img, session, anomaly_treshold = 4000):
        test_samples = np.zeros([32,48,48,3])
        for i, _ in enumerate(test_samples):
            test_samples[i] = test_img
    
        # choose random instances
        ix = randint(0, test_samples.shape[0], self.batch_size)
        # select images
        x_batch = test_samples[ix]
        x_labels = np.zeros(x_batch.shape[0])
        x_recons, z_recons, w_recons, y_recons = self.model_graph.reconstruct_input(session, x_batch, beta=1)
        
        original_x = ((test_img.reshape(48,48,3)*127.5)+127.5).astype(np.uint8)
        similar_img = x_recons[0]
        similar_img = (similar_img*127.5)+127.5
        similar_img = similar_img.astype(np.uint8)
        similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
    
        score = np.sum((original_x.astype("float") - similar_img.astype("float")) ** 2)
        score /= float(original_x.shape[0] * original_x.shape[1])
    
        has_anomaly = 0
        if (score > anomaly_treshold):
            has_anomaly = 1
    
        return (score, has_anomaly)
    
    def reconstruct_eval(self, dataset1, dataset2):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                #print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
            
            progress_bar = Progbar(target=(len(dataset1) + len(dataset2)))
            regular_scores = np.zeros(len(dataset1))
            ano_scores = np.zeros(len(dataset2))

            # Eval the regual data
            for i, img in enumerate(dataset1):
                score, has_anomaly = self.eval_anomaly(img, session, anomaly_treshold=5000)
                regular_scores[i] = score
                progress_bar.update(i, values=[('score', score)])

            # Eval the anomaly data
            for i, img in enumerate(dataset2):
                score, has_anomaly = self.eval_anomaly(img, session, anomaly_treshold=5000)
                ano_scores[i] = score
                progress_bar.update(i + len(dataset1), values=[('score', score)])
        
        return (regular_scores, ano_scores)
    
    def reconstruct_input(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                #print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            #x_batch, x_labels = data.random_batch_with_labels(self.batch_size)
            # choose random instances
            ix = randint(0, data.shape[0], self.batch_size)
            # select images
            x_batch = data[ix]
            x_labels = np.zeros(x_batch.shape[0])
            x_recons, z_recons, w_recons, y_recons = self.model_graph.reconstruct_input(session, x_batch, beta=1)
            
            loss, recons, cond_prior, KL_w, y_prior, L2_loss = self.model_graph.evaluate(session, x_batch)
            valid_string = 'VALID | Loss: ' + str(loss) + \
                            ' | Recons: ' + str(recons) + \
                            ' | CP: ' + str(cond_prior) + \
                            ' | KL_w: ' + str(KL_w) + \
                            ' | KL_y: ' + str(y_prior) + \
                            ' | L2_loss: '+  str(L2_loss)
            #print(valid_string)
            
            return x_batch, x_labels, x_recons, z_recons, w_recons, y_recons 
        
    def generate_embedding(self, data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return
        
            x_batch, x_labels = data.random_batch_with_labels(self.batch_size)
            x_recons, z_recons = self.model_graph.reconstruct_input(session, x_batch, beta=1)
            return x_batch, x_labels, x_recons,  z_recons   
        
    '''  ------------------------------------------------------------------------------
                                         DISTRIBUTIONS
        ------------------------------------------------------------------------------ '''
        
    def print_parameters():
        print('')
            
    