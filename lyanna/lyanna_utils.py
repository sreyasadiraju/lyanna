#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from numpy.fft import irfft, rfft


# For ChainConsumer posterior chains
rel_T0_string    = '$(T_0 - \hat{T_0}) / \hat{T_0}$'
rel_gamma_string = '$(\gamma - \hat{\gamma}) / \hat{\gamma}$'



def rescale_T0(T0s, mode = 'down'):
    if mode == 'down':
        return (T0s - 10500)/5000
    elif mode == 'up':
        return 5000*T0s + 10500
    else:
        raise ValueError('Invalid mode!')

        
def rescale_gamma(gammas, mode = 'down'):
    if mode == 'down':
        return (gammas - 1.48)/0.2
    elif mode == 'up':
        return 0.2*gammas + 1.48
    else:
        raise ValueError('Invalid mode!')
        
        
def rescale_chain(chain, mode = 'down'):
    if chain.shape[0]==2:
        return np.array([rescale_T0(chain[0, :], mode = mode), rescale_gamma(chain[1, :], mode = mode)])
    elif chain.shape[1]==2:
        return np.array([rescale_T0(chain[:, 0], mode = mode), rescale_gamma(chain[:, 1], mode = mode)]).T
    
    
class LRAdapter:
    
    def __init__(self, learning_rate, cooling_rate = 0.02, freq_epoch = 10, start_cooling_epoch = 10, settle_to = 5e-4, sig_tightness = 1, decay_speed = 1):
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.cooling_rate  = cooling_rate
        self.freq_epoch    = freq_epoch
        self.settle_to     = settle_to
        self.start_cooling_epoch = start_cooling_epoch
        self.sig_tightness = sig_tightness
        self.decay_speed   = decay_speed
        
    def cool_adapt(self, epoch):
        if epoch >= self.start_cooling_epoch:
            if epoch%self.freq_epoch == 0:
                self.learning_rate *= (1 - self.cooling_rate)
            #print(self.learning_rate)
            return self.learning_rate 
        else:
            return self.init_learning_rate
    
    def decay_adapt(self, epoch):
        if epoch >= self.start_cooling_epoch:
            self.learning_rate = self.settle_to + (self.init_learning_rate - self.settle_to)/(self.decay_speed*(epoch - self.start_cooling_epoch) + 1)
            return self.learning_rate 
        else:
            return self.init_learning_rate
    
    def sigmoid_adapt(self, epoch):
        self.learning_rate = self.settle_to + (self.init_learning_rate - self.settle_to)/(np.exp(self.sig_tightness*(epoch - self.start_cooling_epoch)) + 1)
        return self.learning_rate
    
    
    
def log_likelihood(data, model, C_inv):
    delta = data - model
    #print(delta.shape)
    return -delta.dot(C_inv.dot(delta))/2.


def log_likelihood_model_dependent_C(data, model, C_inv):
    delta = data - model
    #print(delta.shape)
    s, logdet = np.linalg.slogdet(C_inv)
    if s <= 0: raise ValueError("Invalid value encountered in log_likelihood_model_dependent_C: C_inv has non-positive determinant!")
    else: return -delta.dot(C_inv.dot(delta))/2. + logdet/2.


v0 = np.array([-0.85902894, -0.51192703])
v1 = np.array([-0.51192703, 0.85902894])
def change_basis(x, y):
    a = v0[0]*x + v0[1]*y
    b = v1[0]*x + v1[1]*y
    return a, b


def discard_large_k_modes(spectra, cutoff_k_idx = 257):
    fft_skewers                   = rfft(spectra)
    fft_skewers[:, cutoff_k_idx:] = 0j
    irfft_skewers                 = irfft(fft_skewers)
    return irfft_skewers

def discard_large_k_modes_and_smooth(spectra, k, cutoff_k_idx = 257, R_FWHM = 10820.21):
    fft_skewers = rfft(spectra)
    v_sigma     = 2.998E5 / R_FWHM / 2.35482
    kernel      = np.exp(-v_sigma**2 * k**2 / 2.)
    fft_skewers*= kernel
    fft_skewers[:, cutoff_k_idx:] = 0j
    irfft_skewers                 = irfft(fft_skewers)
    return irfft_skewers


def discard_large_k_modes_and_downsample(spectra, cutoff_k_idx = 257):
    fft_skewers   = rfft(spectra)
    fft_skewers   = fft_skewers[:, :cutoff_k_idx]/8
    irfft_skewers = irfft(fft_skewers)
    return irfft_skewers


def extract_subset(full_dataset, N_files, full_size_each, subset_size_each):
    subset = np.zeros(full_dataset.shape)[:subset_size_each*N_files, :]
    for i in range(N_files):
        subset[i*subset_size_each : (i+1)*subset_size_each, :] = full_dataset[i*full_size_each : i*full_size_each + subset_size_each, :]
    return subset


def FromCholeskyToCovariance(c1s, c2s, c3s):
    try:
        Sigma_matrices = np.zeros((len(c1s), 2, 2))
    except TypeError:
        Sigma_matrices = np.zeros((1, 2, 2))
    Sigma_matrices[:,0,0] = (c2s**2 + c3s**2)/(c1s*c2s)**2
    Sigma_matrices[:,0,1] = -c3s/(c1s*c2s**2)
    Sigma_matrices[:,1,0] = -c3s/(c1s*c2s**2)
    Sigma_matrices[:,1,1] = 1/c2s**2
    return Sigma_matrices


def FromCholeskyToPrecision(c1s, c2s, c3s):
    try:
        Precision_matrices = np.zeros((len(c1s), 2, 2))
    except TypeError:
        Precision_matrices = np.zeros((1, 2, 2))
    Precision_matrices[:,0,0] = c1s**2
    Precision_matrices[:,0,1] = c1s*c3s
    Precision_matrices[:,1,0] = c1s*c3s
    Precision_matrices[:,1,1] = c2s**2 + c3s**2
    return Precision_matrices