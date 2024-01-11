#! /usr/bin/env python

# import numpy as np
import tensorflow as tf 
from lyanna.lyanna_utils import * 



class ResBlock:
    # impl_id 0-
    def __init__(self, num_filters, kernel_size, strides = 2, activation = None, padding = 'same', regularizer = None, initializer = None, dropout = 0.0, idx = 0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides     = strides 
        self.activation  = activation
        self.padding     = padding
        self.regularizer = regularizer
        self.initializer = initializer
        self.dropout     = dropout
        self.idx         = idx 
        self.num_clayers = 3
        
    def __call__(self, x):
        x = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, strides = self.strides, activation = None, padding = self.padding, kernel_regularizer = self.regularizer, kernel_initializer = self.initializer, name = f'conv_{self.num_clayers*self.idx}')(x)
        x_skip = x
        x = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, strides = 1, activation = self.activation, padding = self.padding, kernel_regularizer = self.regularizer, kernel_initializer = self.initializer, name = f'conv_{self.num_clayers*self.idx+1}')(x)
        x = tf.keras.layers.Conv1D(self.num_filters, self.kernel_size, strides = 1, activation = None, padding = self.padding, kernel_regularizer = self.regularizer, kernel_initializer = self.initializer, name = f'conv_{self.num_clayers*self.idx+2}')(x)
        x = tf.keras.layers.Add(name = f'skip_{self.idx}')([x_skip, x])
        x = tf.keras.layers.LeakyReLU()(x)
        return x




def create_rSansa():
    
    conv_dropout_rate  = 0.14
    
    l2_conv  = 2.0732014674545065e-07 
    l2_dense = 3.642709112553934e-08
    
    conv_regularizer   = tf.keras.regularizers.L2(l2 = l2_conv)
    dense_regularizer  = tf.keras.regularizers.L2(l2 = l2_dense)
    
    conv_initializer   = tf.keras.initializers.GlorotNormal(seed = 0)
    dense_initializer  = tf.keras.initializers.HeNormal(seed = 0)
    
    tf.random.set_seed(15)
    # model = tf.keras.Sequential()
    
    inputs = tf.keras.layers.Input(shape = (512,1))
    
    # First Residual Block
    x = ResBlock(16, 16, strides = 2, activation = 'leaky_relu', padding = 'same', regularizer = conv_regularizer, initializer = conv_initializer, dropout = conv_dropout_rate, idx = 0)(inputs)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_0')(x)
    x = tf.keras.layers.Dropout(conv_dropout_rate, name = 'dropout_conv_0')(x)
    x = tf.keras.layers.AveragePooling1D(4, padding = 'same', name = 'avgpool_0')(x)
    
    # Second Residual Block
    x = ResBlock(32, 8, strides = 2, activation = 'leaky_relu', padding = 'same', regularizer = conv_regularizer, initializer = conv_initializer, dropout = conv_dropout_rate, idx = 1)(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_1')(x)
    x = tf.keras.layers.Dropout(conv_dropout_rate, name = 'dropout_conv_1')(x)
    x = tf.keras.layers.AveragePooling1D(4, padding = 'same', name = 'avgpool_1')(x)

    # Third Residual Block
    x = ResBlock(32, 8, strides = 1, activation = 'leaky_relu', padding = 'same', regularizer = conv_regularizer, initializer = conv_initializer, dropout = conv_dropout_rate, idx = 2)(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_2')(x)
    x = tf.keras.layers.Dropout(conv_dropout_rate, name = 'dropout_conv_2')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_2')(x)

    # Fourth Residual Block
    x = ResBlock(64, 8, strides = 1, activation = 'leaky_relu', padding = 'same', regularizer = conv_regularizer, initializer = conv_initializer, dropout = conv_dropout_rate, idx = 3)(x)
    x = tf.keras.layers.BatchNormalization(name = 'batch_norm_3')(x)
    x = tf.keras.layers.Dropout(conv_dropout_rate, name = 'dropout_conv_3')(x)
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same', name = 'avgpool_3')(x)
    
    x = tf.keras.layers.Flatten(name = 'flatten')(x)
    
    output = tf.keras.layers.Dense(5, activation = None, use_bias = False, kernel_regularizer = dense_regularizer, kernel_initializer = dense_initializer, name = 'dense_final')(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = output, name = 'rSansa')
    ## Model building API ends 
    
    return model