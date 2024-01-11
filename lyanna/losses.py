#! /usr/bin/env python

import tensorflow as tf


class NL3_2P(tf.keras.losses.Loss):
    """
    This class is an implementation of the negative log-likelihood loss function. The inverse covariance matrix is decomposed into Cholesky lower-triangular matrix L and its transpose. The network outputs a 5-size vector y with y[0,1] are the point estimates and y[2] = L00, y[3] = L11, y[4] = L10. 
    """    
    def call(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return -tf.math.log(tf.math.square(y_pred[2]*y_pred[3])) + tf.math.square(y_pred[2]*delta_1) + 2.*y_pred[2]*y_pred[4]*delta_1*delta_2 + tf.math.square(y_pred[3]*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def MeanOverallErrorT0(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[0] - y_pred[0]
    
    def MeanOverallErrorGamma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[1] - y_pred[1]
    
    def DeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return 1./tf.math.square(y_pred[2]*y_pred[3])
    
    def LogDeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return -tf.math.log(tf.math.square(y_pred[2]*y_pred[3]))
    
    def MeanSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.math.square(y_true[0] - y_pred[0]) + tf.math.square(y_true[1] - y_pred[1])

    def ChiSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3andMSE(self, y_true, y_pred):
        return self.call(y_true, y_pred) + self.MeanSquaredError(y_true, y_pred)
    
    

class NL3_2P_positive(tf.keras.losses.Loss):
    """
    This class is an implementation of the negative log-likelihood loss function. The inverse covariance matrix is decomposed into Cholesky lower-triangular matrix L and its transpose. The network outputs a 5-size vector y with y[0,1] are the point estimates and y[2] = log(L00), y[3] = log(L11), y[4] = L10 such that the diagonal entries of L are always positive. 
    """
    def call(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return -2.*(y_pred[2] + y_pred[3]) + tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def MeanOverallErrorT0(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[0] - y_pred[0]
    
    def MeanOverallErrorGamma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return y_true[1] - y_pred[1]
    
    def DeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.exp(-2.*(y_pred[2] + y_pred[3]))
    
    def LogDeterminantSigma(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return -2.*(y_pred[2] + y_pred[3])
    
    def MeanSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        return tf.math.square(y_true[0] - y_pred[0]) + tf.math.square(y_true[1] - y_pred[1])

    def ChiSquaredError(self, y_true, y_pred):
        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        delta_1 = y_true[0] - y_pred[0]
        delta_2 = y_true[1] - y_pred[1]
        return tf.math.square(tf.exp(y_pred[2])*delta_1) + 2.*tf.exp(y_pred[2])*y_pred[4]*delta_1*delta_2 + tf.math.square(tf.exp(y_pred[3])*delta_2) + tf.math.square(y_pred[4]*delta_2)
    
    def NL3andMSE(self, y_true, y_pred):
        return self.call(y_true, y_pred) + self.MeanSquaredError(y_true, y_pred)
    
    def Cost(self, y_true, y_pred):
        return self.call(y_true, y_pred) + tf.abs(self.ChiSquaredError(y_true, y_pred) - 2.)