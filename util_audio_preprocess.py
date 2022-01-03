import os
import sys
import tensorflow as tf


def tf_demean(x, axis=1):
    '''
    Helper function to mean-subtract tensor.
    
    Args
    ----
    x (tensor): tensor to be mean-subtracted
    axis (int): kwarg for tf.reduce_mean (axis along which to compute mean)
    
    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
    x_demean = tf.math.subtract(x, tf.reduce_mean(x, axis=1, keepdims=True))
    return x_demean


def tf_rms(x, axis=1, keepdims=True):
    '''
    Helper function to compute RMS amplitude of a tensor.
    
    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    axis (int): kwarg for tf.reduce_mean (axis along which to compute mean)
    keepdims (bool): kwarg for tf.reduce_mean (specify if mean should keep collapsed dimension) 
    
    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
    rms_x = tf.sqrt(tf.reduce_mean(tf.math.square(x), axis=axis, keepdims=keepdims))
    return rms_x


def tf_set_snr(signal, noise, snr):
    '''
    Helper function to combine signal and noise tensors with specified SNR.
    
    Args
    ----
    signal (tensor): signal tensor
    noise (tensor): noise tensor
    snr (tensor): desired signal-to-noise ratio in dB
    
    Returns
    -------
    signal_in_noise (tensor): equal to signal + noise_scaled
    signal (tensor): mean-subtracted version of input signal tensor
    noise_scaled (tensor): mean-subtracted and scaled version of input noise tensor
    
    Raises
    ------
    InvalidArgumentError: Raised when rms(signal) == 0 or rms(noise) == 0.
        Occurs if noise or signal input are all zeros, which is incompatible with set_snr implementation.
    '''
    # Mean-subtract the provided signal and noise
    signal = tf_demean(signal, axis=1)
    noise = tf_demean(noise, axis=1)
    # Compute RMS amplitudes of provided signal and noise
    rms_signal = tf_rms(signal, axis=1, keepdims=True)
    rms_noise = tf_rms(noise, axis=1, keepdims=True)
    # Ensure neither signal nor noise has an RMS amplitude of zero
    msg = 'The rms({:s}) == 0. Results from {:s} input values all equal to zero'
    tf.debugging.assert_none_equal(rms_signal, tf.zeros_like(rms_signal),
                                   message=msg.format('signal','signal')).mark_used()
    tf.debugging.assert_none_equal(rms_noise, tf.zeros_like(rms_noise),
                                   message=msg.format('noise','noise')).mark_used()
    # Convert snr from dB to desired ratio of RMS(signal) to RMS(noise)
    rms_ratio = tf.math.pow(10.0, snr / 20.0)
    # Re-scale RMS of the noise such that signal + noise will have desired SNR
    noise_scale_factor = tf.math.divide(rms_signal, tf.math.multiply(rms_noise, rms_ratio))
    noise_scaled = tf.math.multiply(noise_scale_factor, noise)
    signal_in_noise = tf.math.add(signal, noise_scaled)
    return signal_in_noise, signal, noise_scaled


def tf_set_dbspl(x, dbspl):
    '''
    Helper function to scale tensor to a specified sound pressure level
    in dB re 20e-6 Pa (dB SPL).
    
    Args
    ----
    x (tensor): tensor to be scaled
    dbspl (tensor): desired sound pressure level in dB re 20e-6 Pa
    
    Returns
    -------
    x (tensor): mean-subtracted and scaled tensor
    scale_factor (tensor): constant x is multiplied by to set dB SPL 
    '''
    x = tf_demean(x, axis=1)
    rms_new = 20e-6 * tf.math.pow(10.0, dbspl / 20.0)
    rms_old = tf_rms(x, axis=1, keepdims=True)
    scale_factor = rms_new / rms_old
    x = tf.math.multiply(scale_factor, x)
    return x, scale_factor
