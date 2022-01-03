import os
import sys
import glob
import json
import numpy as np
import tensorflow as tf

import util_recognition_network
import util_cochlear_model


class AuditoryModelLoss():
    def __init__(self,
                 dir_recognition_networks='models/recognition_networks',
                 list_recognition_networks=None,
                 fn_weights='deep_feature_loss_weights.json',
                 config_cochlear_model={},
                 tensor_wave0=None,
                 tensor_wave1=None):
        """
        The AuditoryModelLoss class creates an object to measure distances between
        auditory model representations of sounds.
        
        Args
        ----
        dir_recognition_networks (str): directory containing recognition network architectures
            and model checkpoints
        list_recognition_networks (list or None): list of recognition network keys specifying
            the deep feature loss (if None, use all checkpoints in `dir_recognition_networks`)
        fn_weights (str): filename for deep feature loss weights, which weight the contribution
            of each recognition network layer to the total deep feature loss weight
        config_cochlear_model (dict): parameters for util_cochlear_model.build_cochlear_model
        tensor_wave0 (tensor with shape [batch, 40000]): reference waveform tensor (for training)
        tensor_wave1 (tensor with shape [batch, 40000]): denoised waveform tensor (for training)
        """
        if not os.path.isabs(fn_weights):
            fn_weights = os.path.join(dir_recognition_networks, fn_weights)
        with open(fn_weights, 'r') as f_weights:
            deep_feature_loss_weights = json.load(f_weights)
        if list_recognition_networks is None:
            print(("`list_recognition_networks` not specified --> "
                   "searching for all checkpoints in {}".format(dir_recognition_networks)))
            list_fn_ckpt = glob.glob(os.path.join(dir_recognition_networks, '*index'))
            list_fn_ckpt = [fn_ckpt.replace('.index', '') for fn_ckpt in list_fn_ckpt]
        else:
            list_fn_ckpt = []
            for network_key in list_recognition_networks: 
                tmp = glob.glob(os.path.join(dir_recognition_networks, '{}*index'.format(network_key)))
                msg = "Failed to find exactly 1 checkpoint for recognition network {}".format(network_key)
                assert len(tmp) == 1, msg
                list_fn_ckpt.append(tmp[0].replace('.index', ''))
        print("{} recognition networks included for deep feature loss:".format(len(list_fn_ckpt)))
        config_recognition_networks = {}
        for fn_ckpt in list_fn_ckpt:
            network_key = os.path.basename(fn_ckpt).split('.')[0]
            if 'taskA' in network_key:
                n_classes_dict = {"task_audioset": 517}
            else:
                n_classes_dict = {"task_word": 794}
            config_recognition_networks[network_key] = {
                'fn_ckpt': fn_ckpt,
                'fn_arch': fn_ckpt[:fn_ckpt.rfind('_task')] + '.json',
                'n_classes_dict': n_classes_dict,
                'weights': deep_feature_loss_weights[network_key],
            }
            print('|__ {}: {}'.format(network_key, fn_ckpt))
        self.config_recognition_networks = config_recognition_networks
        self.config_cochlear_model = config_cochlear_model
        self.tensor_wave0 = tensor_wave0
        self.tensor_wave1 = tensor_wave1
        self.build_auditory_model()
        self.sess = None
        self.vars_loaded = False
        return


    def l1_distance(self, feature0, feature1):
        """
        Computes L1 distance between two features (preserving axis 0 for batch)
        """
        axis = np.arange(1, len(feature0.get_shape().as_list()))
        return tf.reduce_sum(tf.math.abs(feature0 - feature1), axis=axis)


    def build_auditory_model(self, dtype=tf.float32):
        """
        Constructs the full auditory model and losses. This function builds two
        waveform placeholders and constructs two identical auditory models operating
        on the two placeholders. Waveform, cochlear model, and deep feature losses are
        computed by measuring distances between identical stages of the two auditory
        models.
        """
        # Build placeholders for two waveforms (if needed) and compute waveform loss
        if (self.tensor_wave0 is None) or (self.tensor_wave1 is None):
            self.tensor_wave0 = tf.placeholder(dtype, [None, 40000])
            self.tensor_wave1 = tf.placeholder(dtype, [None, 40000])
        print('Building waveform loss')
        self.loss_waveform = self.l1_distance(self.tensor_wave0, self.tensor_wave1)
        # Build cochlear model for each waveform and compute cochlear model loss
        print('Building cochlear model loss')
        tensor_coch0, _ = util_cochlear_model.build_cochlear_model(
            self.tensor_wave0,
            **self.config_cochlear_model)
        tensor_coch1, _ = util_cochlear_model.build_cochlear_model(
            self.tensor_wave1,
            **self.config_cochlear_model)
        self.loss_cochlear_model = self.l1_distance(tensor_coch0, tensor_coch1)
        # Build network(s) for each waveform and compute deep feature losses
        self.loss_deep_features_dict = {}
        self.loss_deep_features = tf.zeros([], dtype=dtype)
        for network_key in sorted(self.config_recognition_networks.keys()):
            print('Building deep feature loss (recognition network: {})'.format(network_key))
            with open(self.config_recognition_networks[network_key]['fn_arch'], 'r') as f:
                list_layer_dict = json.load(f)
            # Build network for stimulus 0
            with tf.variable_scope(network_key + '0') as scope:
                _, tensors_network0 = util_recognition_network.build_network(
                    tensor_coch0,
                    list_layer_dict,
                    n_classes_dict=self.config_recognition_networks[network_key]['n_classes_dict'])
                var_list = {
                    v.name.replace(scope.name + '/', '').replace(':0', ''): v
                    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                }
                self.config_recognition_networks[network_key]['saver0'] = tf.train.Saver(
                    var_list=var_list,
                    max_to_keep=0)
            # Build network for stimulus 1
            with tf.variable_scope(network_key + '1') as scope:
                _, tensors_network1 = util_recognition_network.build_network(
                    tensor_coch1,
                    list_layer_dict,
                    n_classes_dict=self.config_recognition_networks[network_key]['n_classes_dict'])
                var_list = {
                    v.name.replace(scope.name + '/', '').replace(':0', ''): v
                    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                }
                self.config_recognition_networks[network_key]['saver1'] = tf.train.Saver(
                    var_list=var_list,
                    max_to_keep=0)
            # Compute deep feature losses (weighted sum across layers)
            self.loss_deep_features_dict[network_key] = tf.zeros([], dtype=dtype)
            layer_weights = self.config_recognition_networks[network_key]['weights']
            for layer_key in sorted(layer_weights.keys()):
                tmp = self.l1_distance(tensors_network0[layer_key], tensors_network1[layer_key])
                self.loss_deep_features_dict[network_key] += layer_weights[layer_key] * tmp
            self.loss_deep_features += self.loss_deep_features_dict[network_key]


    def load_auditory_model_vars(self, sess):
        """
        Loads the same variables into the two copies of each recognition network
        from recognition network checkpoints.
        """
        self.sess = sess
        for network_key in sorted(self.config_recognition_networks.keys()):
            fn_ckpt = self.config_recognition_networks[network_key]['fn_ckpt']
            saver0 = self.config_recognition_networks[network_key]['saver0']
            saver1 = self.config_recognition_networks[network_key]['saver1']
            print('Loading `{}` variables from {}'.format(network_key, fn_ckpt))
            saver0.restore(self.sess, fn_ckpt)
            saver1.restore(self.sess, fn_ckpt)
        self.vars_loaded = True


    def waveform_loss(self, y0, y1):
        """
        Method to compute waveform loss between two waveforms under
        the constructed auditory model.
        
        Args
        ----
        y0 (np.ndarray): waveform with shape [batch, 40000]
        y1 (np.ndarray): waveform with shape [batch, 40000]
        
        Returns
        -------
        (np.ndarray): L1 waveform loss with shape [batch]
        """
        assert (self.sess is not None) and (not self.sess._closed)
        feed_dict={self.tensor_wave0: y0, self.tensor_wave1: y1}
        return self.sess.run(self.loss_waveform, feed_dict=feed_dict)


    def cochlear_model_loss(self, y0, y1):
        """
        Method to compute cochlear model loss between two waveforms
        under the constructed auditory model.
        
        Args
        ----
        y0 (np.ndarray): waveform with shape [batch, 40000]
        y1 (np.ndarray): waveform with shape [batch, 40000]
        
        Returns
        -------
        (np.ndarray): L1 cochlear model loss with shape [batch]
        """
        assert (self.sess is not None) and (not self.sess._closed)
        feed_dict={self.tensor_wave0: y0, self.tensor_wave1: y1}
        return self.sess.run(self.loss_cochlear_model, feed_dict=feed_dict)


    def deep_feature_loss(self, y0, y1):
        """
        Method to compute deep feature loss between two waveforms
        under the constructed auditory model.
        
        Args
        ----
        y0 (np.ndarray): waveform with shape [batch, 40000]
        y1 (np.ndarray): waveform with shape [batch, 40000]
        
        Returns
        -------
        (np.ndarray): L1 deep feature loss with shape [batch]
        """
        assert (self.sess is not None) and (not self.sess._closed)
        if not self.vars_loaded:
            print(("WARNING: `deep_feature_loss` called before loading vars"))
        feed_dict={self.tensor_wave0: y0, self.tensor_wave1: y1}
        return self.sess.run(self.loss_deep_features, feed_dict=feed_dict)
