import os
import sys

import tensorflow as tf

sys.path.append('Wave-U-Net')
import UnetAudioSeparator


def build_unet(tensor_waveform, signal_rate=20000, UNET_PARAMS={}):
    '''
    This function builds the tensorflow graph for the Wave-U-Net.
    
    Args
    ----
    tensor_waveform (tensor): input audio waveform (shape: [batch, time])
    signal_rate (int): sampling rate of input waveform (Hz)
    UNET_PARAMS (dict): U-net configuration parameters
    
    Returns
    -------
    tensor_waveform_unet (tensor): U-net transformed waveform (shape: [batch, time])
    '''
    padding = UNET_PARAMS.get('padding', [[0,0], [480,480]])
    tensor_waveform_zero_padded = tf.pad(tensor_waveform, padding)
    tensor_waveform_expanded = tf.expand_dims(tensor_waveform_zero_padded,axis=2)
    unet_audio_separator = UnetAudioSeparator.UnetAudioSeparator(UNET_PARAMS)
    unet_audio_separator_output = unet_audio_separator.get_output(
        tensor_waveform_expanded,
        training=True,
        return_spectrogram=False,
        reuse=tf.AUTO_REUSE)
    tensor_waveform_unet = unet_audio_separator_output["enhancement"]
    tensor_waveform_unet = tensor_waveform_unet[:, padding[1][0]:-padding[1][1],:]
    tensor_waveform_unet = tf.squeeze(tensor_waveform_unet, axis=2)
    return tensor_waveform_unet
