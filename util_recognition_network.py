import os
import sys
import numpy as np
import tensorflow as tf


def build_network(tensor_input, list_layer_dict, n_classes_dict={}):
    """
    Build tensorflow graph for a feedforward neural network given an
    input tensor and list of layer descriptions
    """
    tensor_output = tensor_input
    tensors_dict = {}
    for layer_dict in list_layer_dict:
        layer_type = layer_dict['layer_type']
        if 'batch_normalization' in layer_type:
            layer = tf.keras.layers.BatchNormalization(**layer_dict['args'])
        elif 'conv2d' in layer_type:
            layer = PaddedConv2D(**layer_dict['args'])
        elif 'dense' in layer_type:
            layer = tf.keras.layers.Dense(**layer_dict['args'])
        elif 'dropout' in layer_type:
            layer = tf.keras.layers.Dropout(**layer_dict['args'])
        elif 'hpool' in layer_type:
            layer = HanningPooling(**layer_dict['args'])
        elif 'flatten' in layer_type:
            layer = tf.keras.layers.Flatten(**layer_dict['args'])
        elif 'leaky_relu' in layer_type:
            layer = tf.keras.layers.LeakyReLU(**layer_dict['args'])
        elif 'relu' in layer_type:
            layer = tf.keras.layers.ReLU(**layer_dict['args'])
        elif 'multi_fc_top' in layer_type:
            layer = LegacyDenseTaskHeads(
                n_classes_dict=n_classes_dict,
                **layer_dict['args'])
        elif 'fc_top' in layer_type:
            layer = DenseTaskHeads(
                n_classes_dict=n_classes_dict,
                **layer_dict['args'])
        else:
            msg = "layer_type={} not recognized".format(layer_type)
            raise NotImplementedError(msg)
        tensor_output = layer(tensor_output)
        if layer_dict.get('args', {}).get('name', None) is not None:
            tensors_dict[layer_dict['args']['name']] = tensor_output
    return tensor_output, tensors_dict


def same_pad_along_axis(tensor_input,
                        kernel_length=1,
                        stride_length=1,
                        axis=1,
                        **kwargs):
    """
    Adds 'SAME' padding to only specified axis of tensor_input
    for 2D convolution
    """
    x = tensor_input.shape.as_list()[axis]
    if x % stride_length == 0:
        p = kernel_length - stride_length
    else:
        p = kernel_length - (x % stride_length)
    p = tf.math.maximum(p, 0)
    paddings = [(0, 0)] * len(tensor_input.shape)
    paddings[axis] = (p // 2, p - p // 2)
    return tf.pad(tensor_input, paddings, **kwargs)


def PaddedConv2D(filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 **kwargs):
    """
    Wrapper function around tf.keras.layers.Conv2D to support
    custom padding options
    """
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=kernel_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    def layer(tensor_input):
        conv2d_layer = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            **kwargs)
        return conv2d_layer(pad_function(tensor_input))
    return layer


def HanningPooling(strides=2,
                   pool_size=8,
                   padding='SAME',
                   sqrt_window=False,
                   normalize=False,
                   name=None):
    """
    Weighted average pooling layer with Hanning window applied via
    2D convolution (with identity transform as depthwise component)
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    assert len(strides) == 2, "HanningPooling expects 2D args"
    assert len(pool_size) == 2, "HanningPooling expects 2D args"
    
    (dim0, dim1) = pool_size
    if dim0 == 1:
        win0 = np.ones(dim0)
    else:
        win0 = (1 - np.cos(2 * np.pi * np.arange(dim0) / (dim0 - 1))) / 2
    if dim1 == 1:
        win1 = np.ones(dim1)
    else:
        win1 = (1 - np.cos(2 * np.pi * np.arange(dim1) / (dim1 - 1))) / 2
    hanning_window = np.outer(win0, win1)
    if sqrt_window:
        hanning_window = np.sqrt(hanning_window)
    if normalize:
        hanning_window = hanning_window / hanning_window.sum()
    
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=pool_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    
    def layer(tensor_input):
        tensor_hanning_window = tf.constant(
            hanning_window[:, :, np.newaxis, np.newaxis],
            dtype=tensor_input.dtype,
            name="{}_hanning_window".format(name))
        tensor_eye = tf.eye(
            num_rows=tensor_input.shape.as_list()[-1],
            num_columns=None,
            batch_shape=[1, 1],
            dtype=tensor_input.dtype,
            name=None)
        tensor_output = tf.nn.convolution(
            pad_function(tensor_input),
            tensor_hanning_window * tensor_eye,
            strides=strides,
            padding=padding,
            data_format=None,
            name=name)
        return tensor_output
    
    return layer


def DenseTaskHeads(n_classes_dict={}, name='logits', **kwargs):
    """
    Dense layer for each task head specified in n_classes_dict
    """
    def layer(tensor_input):
        tensors_logits = {}
        for key in sorted(n_classes_dict.keys()):
            if len(n_classes_dict.keys()) > 1:
                classification_name = '{}_{}'.format(name, key)
            else:
                classification_name = name
            classification_layer = tf.keras.layers.Dense(
                units=n_classes_dict[key],
                name=classification_name,
                **kwargs)
            tensors_logits[key] = classification_layer(tensor_input)
        return tensors_logits
    return layer


def LegacyDenseTaskHeads(n_classes_dict={}, name='logits', **kwargs):
    """
    Legacy code
    """
    def layer(tensor_input):
        tensors_logits = {}
        for idx, key in enumerate(sorted(n_classes_dict.keys())):
            classification_name = '{}_{}'.format(name, idx)
            classification_layer = tf.keras.layers.Dense(
                units=n_classes_dict[key],
                name=classification_name,
                **kwargs)
            tensors_logits[key] = classification_layer(tensor_input)
        return tensors_logits
    return layer
