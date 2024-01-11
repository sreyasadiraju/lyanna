#! /usr/bin/env python

import h5py
import numpy as np
from glob import glob
import tensorflow as tf
from lyanna.lyanna_utils import *


class FilesOracle:
    
    def __init__(self, hfilespath):
        self.hfilespath = hfilespath 
        hfiles          = glob(hfilespath + '*.h5')

        self.dtype = np.dtype([('T0', 'f8'), ('gamma', 'f8'), ('alpha', 'f8'), ('beta', 'f8'), ('filename', 'U300')])

        Oracle     = []
        for i, file in enumerate(hfiles):
            with h5py.File(file, 'r') as f:
                T0, gamma = f['tdr_params'][:]
            T0        = rescale_T0(T0) 
            gamma     = rescale_gamma(gamma)
            alpha, beta = change_basis(T0, gamma)
            Oracle.append((T0, gamma, np.round(alpha,8), np.round(beta,8), file))
            
        Oracle = np.array(Oracle, dtype = self.dtype)
        Oracle = np.sort(Oracle, order = ['alpha', 'beta'])
        
        self.Oracle = Oracle
        self.T0s    = Oracle['T0']
        self.gammas = Oracle['gamma']
        self.hfiles = Oracle['filename']
        self.alphas = Oracle['alpha']
        self.betas  = Oracle['beta']

        with h5py.File(self.hfiles[0]) as f:
            self.k_full    = f['k_in_skminv'][1:]
            self.v_h_skewer_full = f['v_h_skewer'][:]
        self.k             = self.k_full[:256]
        self.N_pixels_full = len(self.v_h_skewer_full)
        self.delta_v_full  = abs(self.v_h_skewer_full[1] - self.v_h_skewer_full[0])/1e5


    def in_prior_volume(self, T0, gamma):
        alpha, beta = change_basis(T0, gamma)
        if alpha < self.alphas.max()-0.01 and alpha > self.alphas.min()+0.01 and beta < self.betas.max()-0.01 and beta > self.betas.min()+0.01:
            return True
        else:
            return False


def extract_subset(full_dataset, N_files, full_size_each, subset_size_each):
    subset = np.zeros(full_dataset.shape)[:subset_size_each*N_files, :]
    for i in range(N_files):
        subset[i*subset_size_each : (i+1)*subset_size_each, :] = full_dataset[i*full_size_each : i*full_size_each + subset_size_each, :]
    return subset

def roll_spectral_dataset(spectral_dataset, N_roll = 256, N_pix_spectrum = 512): # spectral_dataset needs to be in lyanna dataset format
    new_set = np.copy(spectral_dataset)
    new_set[:, :N_pix_spectrum] = np.roll(spectral_dataset[:,:N_pix_spectrum], N_roll, axis = 1)
    return new_set

def roll_spectral_dataset_randomly(spectral_dataset, N_pix_spectrum = 512, seed = 100): # spectral_dataset needs to be in lyanna dataset format
    new_set = np.copy(spectral_dataset)
    np.random.seed(seed)
    N_roll_arr = np.random.randint(0, N_pix_spectrum, spectral_dataset.shape[0])
    for i in range(spectral_dataset.shape[0]):
        new_set[i, :N_pix_spectrum] = np.roll(spectral_dataset[i,:N_pix_spectrum], N_roll_arr[i], axis = 0)
    return new_set

def flip_spectral_dataset(spectral_dataset, N_pix_spectrum = 512, randomly = False, seed = 100):
    new_set = np.copy(spectral_dataset)
    if randomly:
        np.random.seed(seed)
        yesno = 2*np.random.randint(0, 2, spectral_dataset.shape[0]) - 1
        for i in range(spectral_dataset.shape[0]):
            new_set[i, :N_pix_spectrum] = spectral_dataset[i,:N_pix_spectrum][::yesno[i],:]
    else:
        new_set[:, :N_pix_spectrum] = spectral_dataset[:,:N_pix_spectrum][:,::-1,:]
    return new_set



class Noise:
    def __init__(self, SNR, v_h_skewer, CNR = True, N_pix_spectrum = 512):
        self.SNR            = SNR
        self.v_h_skewer     = v_h_skewer
        # self.CNR            = CNR
        self.N_pix_spectrum = N_pix_spectrum
        if CNR:
            self.sigv = 1./SNR
        self.R        = (v_h_skewer[1] - v_h_skewer[0]) * (4096/N_pix_spectrum)*1e-5 /6.
        self.sigp     = tf.convert_to_tensor(np.sqrt(self.R)*self.sigv)
        self.seed     = 0
        
    def add_noise(self, spectral_inputs):
        # np.random.seed(self.seed)
        tf.random.set_seed(self.seed);
        noise         = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
        noisy_inputs  = tf.identity(spectral_inputs)
        # noise      = np.random.normal(0.0, self.sigp, spectral_dataset[:, :self.N_pix_spectrum, :].shape)
        noise         = tf.dtypes.cast(noise, tf.float32)
        noisy_inputs  = tf.math.add(noisy_inputs, noise)
        self.seed    += 1
        return noisy_inputs



class Augmentation:
    def __init__(self, p_flip, seed = 0):
        self.p_flip = p_flip
        self.seed   = seed
    

    def init_noise(self, SNR, v_h_skewer, CNR = True, N_pix_spectrum = 512):
        self.SNR            = SNR
        self.v_h_skewer     = v_h_skewer
        # self.CNR            = CNR
        self.N_pix_spectrum = N_pix_spectrum
        if CNR:
            self.sigv = 1./SNR
        self.R        = (v_h_skewer[1] - v_h_skewer[0]) * (4096/N_pix_spectrum)*1e-5 /6.
        self.sigp     = tf.convert_to_tensor(np.sqrt(self.R)*self.sigv)
        # self.seed     = self.seed


    def roll(self, spectral_inputs, random_indices):
        rows, cols, _ = spectral_inputs.shape
        shifts = tf.repeat(tf.expand_dims(random_indices, axis = 1), cols, axis = 1)
        row_indices = tf.tile(tf.expand_dims(tf.range(rows), axis = 1), [1, cols])
        indices = tf.stack([row_indices, (tf.range(cols) - shifts)%cols], axis = -1)
        return tf.gather_nd(spectral_inputs, indices)
    

    def flip(self, spectral_inputs, random_indices):
        return tf.tensor_scatter_nd_update(spectral_inputs, tf.expand_dims(random_indices, axis = 1), tf.reverse(tf.gather(spectral_inputs, random_indices), axis = [1]))
    

    def augment_without_noise(self, spectral_inputs):
        tf.random.set_seed(self.seed)
        I = tf.range(spectral_inputs.shape[0])
        N = int(spectral_inputs.shape[0]*self.p_flip)
        indices_roll = tf.random.uniform(shape=(spectral_inputs.shape[0],), minval=1, maxval=spectral_inputs.shape[1], dtype=tf.int32)
        indices_flip = tf.random.shuffle(I)[:N] #self.RNG_f.choice(a, size = int(spectral_inputs.shape[0]*self.p_flip), replace = False)
        spectra          = self.flip(spectral_inputs, indices_flip)
        spectral_outputs = self.roll(spectra, indices_roll)
        self.seed += 1
        return spectral_outputs
    

    def augment_with_noise(self, spectral_inputs):
        tf.random.set_seed(self.seed)
        I = tf.range(spectral_inputs.shape[0])
        N = int(spectral_inputs.shape[0]*self.p_flip)
        indices_roll = tf.random.uniform(shape=(spectral_inputs.shape[0],), minval=1, maxval=spectral_inputs.shape[1], dtype=tf.int32)
        indices_flip = tf.random.shuffle(I)[:N] #self.RNG_f.choice(a, size = int(spectral_inputs.shape[0]*self.p_flip), replace = False)
        spectra          = self.flip(spectral_inputs, indices_flip)
        spectral_outputs = self.roll(spectra, indices_roll)
        try:
            noise     = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
            noise     = tf.dtypes.cast(noise, tf.float32)
        except NameError:
            raise NotImplementedError("Noise not initialized! Do ClassInstance.init_noise(__) before trying to add noise to the input spectra.")
        noisy_inputs  = tf.identity(spectral_outputs)
        noisy_inputs  = tf.math.add(noisy_inputs, noise)
        self.seed    += 1
        return noisy_inputs
    
    
    def add_noise(self, spectral_inputs):
        tf.random.set_seed(self.seed);
        noise         = tf.random.normal(tf.shape(spectral_inputs), 0, self.sigp, tf.float64)
        noisy_inputs  = tf.identity(spectral_inputs)
        noise         = tf.dtypes.cast(noise, tf.float32)
        noisy_inputs  = tf.math.add(noisy_inputs, noise)
        self.seed    += 1
        return noisy_inputs




def get_data_for_sansa(training = False, validation = False, test = False, N_sk_each_train = 24000, N_sk_each_valid = 8000, N_sk_each_test = 4000, extension = False, smooth = True, extalpha = False, N_extrows_beta = 2):
    filespath = '/project/ls-gruen/users/parth.nayak/lya_synthesis_data/rescaled_tdr_models/orthogonal_grid_models/'
    extpath   = '/project/ls-gruen/users/parth.nayak/lya_synthesis_data/rescaled_tdr_models/ortho_extended_grid_models/'
    extalphapath = '/project/ls-gruen/users/parth.nayak/lya_synthesis_data/rescaled_tdr_models/ortho_extended_alpha_grid_models/'
    files     = glob(filespath+'*.h5'); extfiles = glob(extpath+'*.h5'); extalphafiles = glob(extalphapath+'*.h5')
    total_hfiles = len(files)
    total_extfiles = len(extfiles)
    total_extalphafiles = len(extalphafiles)
    N_sk_each_train_full = 60000; N_sk_each_test_full = 10000

    indices_to_pick  = []
    for indstart in range(3-N_extrows_beta, 3+N_extrows_beta):
        indices_to_pick.append(np.arange(indstart, indstart+total_extfiles, 3*2))
    indices_to_pick  = np.array(indices_to_pick)
    indices_to_pick  = np.ravel(indices_to_pick)
    indices_to_pick  = np.sort(indices_to_pick)

    if smooth:
        filename_suffix = '_spectra_downsampled_kcutoff_smooth.npy'
        extname_suffix  = '_spectra_downsampled_extorthogrid_kcutoff_smooth.npy'
        extalpha_suffix = '_spectra_downsampled_extalpha_kcutoff_smooth.npy'
        N_sk_each_valid_full = 30000
    else:      
        filename_suffix = '_spectra_downsampled_kcutoff.npy'
        extname_suffix  = '_spectra_downsampled_extorthogrid_kcutoff.npy'
        N_sk_each_valid_full = 40000
    data = {}
    if training:
        train_set = np.load(filespath+'train'+filename_suffix)
        train_set = extract_subset(train_set, total_hfiles, N_sk_each_train_full, N_sk_each_train)
        if extension: 
            train_ext = np.load(extpath+'train'+extname_suffix)
            train_ext = extract_subset(train_ext, total_extfiles, N_sk_each_train_full, N_sk_each_train)
            ext = []
            for ind in indices_to_pick:
                ext += list(train_ext[ind*N_sk_each_train : (ind+1)*N_sk_each_train])
            ext = np.array(ext)
            train_set = np.concatenate((train_set, ext), axis = 0)
        if extalpha: 
            train_extalpha = np.load(extalphapath+'train'+extalpha_suffix)
            train_extalpha = extract_subset(train_extalpha, total_extalphafiles, N_sk_each_train_full, N_sk_each_train)
            train_set = np.concatenate((train_set, train_extalpha), axis = 0)
        data['training'] = train_set
    if validation:
        valid_set = np.load(filespath+'valid'+filename_suffix)
        valid_set = extract_subset(valid_set, total_hfiles, N_sk_each_valid_full, N_sk_each_valid)
        if extension: 
            valid_ext = np.load(extpath+'valid'+extname_suffix)
            valid_ext = extract_subset(valid_ext, total_extfiles, N_sk_each_valid_full, N_sk_each_valid)
            ext = []
            for ind in indices_to_pick:
                ext += list(valid_ext[ind*N_sk_each_valid : (ind+1)*N_sk_each_valid])
            ext = np.array(ext)
            valid_set = np.concatenate((valid_set, ext), axis = 0)
        if extalpha: 
            valid_extalpha = np.load(extalphapath+'valid'+extalpha_suffix)
            valid_extalpha = extract_subset(valid_extalpha, total_extalphafiles, N_sk_each_valid_full, N_sk_each_valid)
            valid_set = np.concatenate((valid_set, valid_extalpha), axis = 0)
        data['validation'] = valid_set
    if test:
        test_set  = np.load(filespath+'test'+filename_suffix)
        test_set  = extract_subset(test_set, total_hfiles, N_sk_each_test_full, N_sk_each_test)
        if extension: 
            test_ext = np.load(extpath+'test'+extname_suffix)
            test_ext = extract_subset(test_ext, total_extfiles, N_sk_each_test_full, N_sk_each_test)
            ext = []
            for ind in indices_to_pick:
                ext += list(test_ext[ind*N_sk_each_test : (ind+1)*N_sk_each_test])
            ext = np.array(ext)
            test_set = np.concatenate((test_set, ext), axis = 0)
        if extalpha: 
            test_extalpha = np.load(extalphapath+'test'+extalpha_suffix)
            test_extalpha = extract_subset(test_extalpha, total_extalphafiles, N_sk_each_test_full, N_sk_each_test)
            test_set = np.concatenate((test_set, test_extalpha), axis = 0)
        data['test'] = test_set
    return data 



def sansa_test_data_inference(filepath, N_sk_to_pick, smooth = True):
    test_spectra = np.zeros((N_sk_to_pick, 512+5, 1), dtype = 'float32')
    with h5py.File(filepath, 'r') as f:
        A_z_0  = f['A_tau_rescaling/A_z_0'][0]
        F_sk   = np.exp( -A_z_0 * f['tau'][:N_sk_to_pick, :])
        k_full = f['k_in_skminv'][:]
        T0, gamma = f['tdr_params'][:]
    T0    = rescale_T0(T0)
    gamma = rescale_gamma(gamma)
    if smooth: 
        F_sk  = discard_large_k_modes_and_smooth(F_sk, k_full)
    else: 
        F_sk  = discard_large_k_modes(F_sk)
    F_sk  = np.mean(F_sk.reshape((-1, 512, 8)), axis = -1)
    test_spectra[:, :512, 0]    = F_sk
    test_spectra[:, 512:514, 0] = T0, gamma
    return test_spectra



def sansa_read_data_file(filepath, N_sk_to_pick):
    spectra = np.zeros((N_sk_to_pick, 512+5, 1), dtype = 'float32')
    with h5py.File(filepath, 'r') as f:
        A_z_0 = f['A_tau_rescaling/A_z_0'][0]
        F_sk  = np.exp( -A_z_0 * f['tau'][:N_sk_to_pick, :])
        T0, gamma = f['tdr_params'][:]
        k         = f['k_in_skminv'][:]
    R     = 60000
    T0    = rescale_T0(T0)
    gamma = rescale_gamma(gamma)
    F_sk  = discard_large_k_modes_and_smooth(F_sk, k, R_FWHM = R)
    F_sk  = np.mean(F_sk.reshape((-1, 512, 8)), axis = -1)
    spectra[:, :512, 0]    = F_sk
    spectra[:, 512:514, 0] = T0, gamma
    return spectra