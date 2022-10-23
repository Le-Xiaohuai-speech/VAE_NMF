# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:39:48 2020

@author: lenovo
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as KB
import tensorflow.keras as K

class VAE:
    
    def __init__(self, input_dim=None, latent_dim=None,intermediate_dim = None,batch_size=None, activation=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.activation = activation
        self.intermediate_dim = intermediate_dim
    def build(self):
        
        self.input_encoder = Input(batch_shape=(self.batch_size, self.input_dim), name='input_layer_encoder')
        
        encoder_1 = Dense(self.intermediate_dim,activation=self.activation,name='encoder_1')(self.input_encoder)
        
        self.latent_mean = Dense(self.latent_dim,
                            name='latent_mean')(encoder_1)

        # Output layer of the encoder: logarithm of the latent variable variance (Type: Tensor)
        self.latent_log_var = Dense(self.latent_dim,
                               name='latent_log_var')(encoder_1)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = KB.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,stddev=1.0)
            return z_mean + KB.exp(z_log_var / 2) * epsilon
        # reparameterization
        # Wrap the sampling function as a layer in Keras (Type: Tensor)
        z = Lambda(sampling)([self.latent_mean, self.latent_log_var])
        
        decoder_1 = Dense(self.intermediate_dim,activation=self.activation,name='decoder_1')(z)
        
        self.output_decoder = Dense(self.input_dim,
                               name='output_layer_decoder')(decoder_1)
        self.output_sp_ev = Dense(self.input_dim,
                               name='output_sp_ev') (decoder_1)
        self.output_ap = Dense(self.input_dim,
                               name='output_ap',activation='sigmoid') (decoder_1)
        self.output_f0 = Dense(1,
                               name='output_f0',activation='relu') (decoder_1)
        self.model = Model(inputs=[self.input_encoder],outputs=[self.output_decoder,self.output_sp_ev,self.output_ap,self.output_f0])
        self.last_model = Model(self.input_encoder,self.output_decoder)
        
    def buid_models(self):
        self.Encoder = Model(self.input_encoder,[self.latent_mean,self.latent_log_var])
        decoder_Input = Input(shape=( self.latent_dim,), name='input_layer_decoder')
        h1 = self.model.get_layer('decoder_1')(decoder_Input)
        o1 = self.model.get_layer('output_layer_decoder')(h1)
        o2 = self.model.get_layer('output_sp_ev')(h1)
        o3 = self.model.get_layer('output_ap')(h1)
        o4 = self.model.get_layer('output_f0')(h1)
        self.Decoder_sp = Model(decoder_Input,o1)
        self.Decoder_spe = Model(decoder_Input,o2)
        self.Decoder_ap = Model(decoder_Input,o3)
        self.Decoder_f0 = Model(decoder_Input,o4)
        
    def compile(self,opt ='adam'):
        def rec_loss_fun0(y_true,y_pred):
            return KB.sum(y_pred +y_true/KB.exp(y_pred), axis=-1 )
        def rec_loss_fun1(y_true,y_pred):
            return KB.sum((y_pred -y_true)**2, axis=-1 )
        def rec_loss_fun2(y_true,y_pred):
            kl_loss = - 0.5 * KB.sum(self.model.get_layer('latent_log_var').output
                                - KB.square(self.model.get_layer('latent_mean').output)
                                - KB.exp(self.model.get_layer('latent_log_var').output ), axis=-1)
            return KB.sum((y_pred -y_true)**2, axis=-1 )+kl_loss
        '''
        reconstruction_loss_0 = KB.sum(self.output_decoder +self.input_encoder/KB.exp(self.output_decoder), axis=-1 )
        reconstruction_loss_1 = KB.sum(self.output_sp_ev +self.input_sp/KB.exp(self.output_sp_ev), axis=-1 )
        reconstruction_loss_2 = KB.sum((self.output_ap-self.input_ap)**2,axis = -1)
        reconstruction_loss_3 = KB.sum((self.output_f0-self.input_f0)**2)
        '''
    # Regularization term
        '''
        kl_loss = - 0.5 * KB.sum(self.model.get_layer('latent_log_var').output
                                - KB.square(self.model.get_layer('latent_mean').output)
                                - KB.exp(self.model.get_layer('latent_log_var').output ), axis=-1)
        '''
        #self.loss = reconstruction_loss_0+reconstruction_loss_1+reconstruction_loss_2+reconstruction_loss_3+ kl_loss
        #self.model.add_loss()
        self.model.compile(optimizer=opt,loss={'output_layer_decoder':rec_loss_fun0,
                                               'output_sp_ev':rec_loss_fun0,
                                               'output_ap':rec_loss_fun1,
                                               'output_f0':rec_loss_fun2},
                               loss_weights=  {'output_layer_decoder':0.25,
                                               'output_sp_ev':0.25,
                                               'output_ap':0.25,
                                               'output_f0':1})
     #   KB.set_value()
     
class Encoder:
    def __init__(self,vae):
        self.vae = vae
    def encode(self,input_data):
        z_mean,z_log_var = self.vae.Encoder.predict(input_data)
        return z_mean,z_log_var 
    
class Decoder:
    def __init__(self,vae):
        self.vae = vae
    def decode(self,input_data):
        output_log_sp = self.vae.Decoder_sp.predict(input_data)
        return output_log_sp
    def decode_sys(self,input_data):
        output_log_spe = self.vae.Decoder_spe.predict(input_data)
        output_ap = self.vae.Decoder_ap.predict(input_data)
        output_f0 = self.vae.Decoder_f0.predict(input_data)
        
        return output_log_spe,output_ap,output_f0
        
if __name__ == '__main__':
    
    vae =VAE(513,64,512,32,'tanh')
    vae.build()
    vae.compile()
    vae.model.load_weights('vaemodel.h5')
    vae.buid_models()
   
