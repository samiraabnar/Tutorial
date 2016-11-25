
# coding: utf-8

# In[8]:

import theano
import theano.tensor as T
import numpy as np

from lasagne.updates import sgd,apply_momentum


# In[21]:

class ImageCaptionGeneraot(object):
    
    def __init__(self, 
                 image_dims, 
                 image_rep_dim, 
                 lstm_input_dim, 
                 lstm_hidden_dim, 
                 lstm_output_dim, 
                 learning_rate):
    
        self.image_dims = image_dims
        self.image_rep_dim = image_rep_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.learning_rate = learning_rate
        
        


# In[ ]:



