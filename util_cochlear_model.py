from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import warnings
import functools
import numpy as np
import tensorflow as tf
import scipy.signal as signal
import matplotlib.pyplot as plt


def freq2erb(freq_hz):
    """Converts Hz to human-defined ERBs, using the formula of Glasberg and Moore.

    Args:
        freq_hz (array_like): frequency to use for ERB.

    Returns:
        ndarray: **n_erb** -- Human-defined ERB representation of input.
    """
    return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """Converts human ERBs to Hz, using the formula of Glasberg and Moore.
    Args:
        n_erb (array_like): Human-defined ERB to convert to frequency.
    Returns:
        ndarray: **freq_hz** -- Frequency representation of input.
    """
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def get_freq_rand_conversions(xp, seed=0, minval=0.0, maxval=1.0):
    """Generates freq2rand and rand2freq conversion functions.

    Args:
        xp (array_like): xvals for freq2rand linear interpolation.
        seed (int): numpy seed to generate yvals for linear interpolation.
        minval (float): yvals for linear interpolation are scaled to [minval, maxval].
        maxval (float): yvals for linear interpolation are scaled to [minval, maxval].

    Returns:
        freq2rand (function): converts Hz to random frequency scale
        rand2freq (function): converts random frequency scale to Hz
    """
    np.random.seed(seed)
    yp = np.cumsum(np.random.poisson(size=xp.shape))
    yp = ((maxval - minval) * (yp - yp.min())) / (yp.max() - yp.min()) + minval
    freq2rand = lambda x : np.interp(x, xp, yp)
    rand2freq = lambda y : np.interp(y, yp, xp)
    return freq2rand, rand2freq


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
    """Generate a half-cosine filter. Represents one subband of the cochleagram.
    A half-cosine filter is created using the values of freqs that are within the
    interval [l, h]. The half-cosine filter is centered at the center of this
    interval, i.e., (h - l) / 2. Values outside the valid interval [l, h] are
    discarded. So, if freqs = [1, 2, 3, ... 10], l = 4.5, h = 8, the cosine filter
    will only be defined on the domain [5, 6, 7] and the returned output will only
    contain 3 elements.
    Args:
        freqs (array_like): Array containing the domain of the filter, in ERB space;
            see convert_to_erb parameter below.. A single half-cosine
            filter will be defined only on the valid section of these values;
            specifically, the values between cutoffs ``l`` and ``h``. A half-cosine filter
            centered at (h - l ) / 2 is created on the interval [l, h].
        l (float): The lower cutoff of the half-cosine filter in ERB space; see
            convert_to_erb parameter below.
        h (float): The upper cutoff of the half-cosine filter in ERB space; see
            convert_to_erb parameter below.
        convert_to_erb (bool, default=True): If this is True, the values in
            input arguments ``freqs``, ``l``, and ``h`` will be transformed from Hz to ERB
            space before creating the half-cosine filter. If this is False, the
            input arguments are assumed to be in ERB space.
    Returns:
        ndarray: **half_cos_filter** -- A half-cosine filter defined using elements of
            freqs within [l, h].
    """
    if convert_to_erb:
        freqs_erb = freq2erb(freqs)
        l_erb = freq2erb(l)
        h_erb = freq2erb(h)
    else:
        freqs_erb = freqs
        l_erb = l
        h_erb = h

    avg_in_erb = (l_erb + h_erb) / 2  # center of filter
    rnge_in_erb = h_erb - l_erb  # width of filter
    # return np.cos((freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint
    # return np.cos((freqs_erb[(freqs_erb >= l_erb) & (freqs_erb <= h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval
    return np.cos((freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval


def make_full_filter_set(filts, signal_length=None):
    """Create the full set of filters by extending the filterbank to negative FFT
    frequencies.
    Args:
        filts (array_like): Array containing the cochlear filterbank in frequency space,
            i.e., the output of make_cos_filters_nx. Each row of ``filts`` is a
            single filter, with columns indexing frequency.
        signal_length (int, optional): Length of the signal to be filtered with this filterbank.
            This should be equal to filter length * 2 - 1, i.e., 2*filts.shape[1] - 1, and if
            signal_length is None, this value will be computed with the above formula.
            This parameter might be deprecated later.

    Returns:
        ndarray: **full_filter_set** -- Array containing the complete filterbank in
            frequency space. This output can be directly applied to the frequency
            representation of a signal.
    """
    if signal_length is None:
        signal_length = 2 * filts.shape[1] - 1

    # note that filters are currently such that each ROW is a filter and COLUMN idxs freq
    if np.remainder(signal_length, 2) == 0:  # even -- don't take the DC & don't double sample nyquist
        neg_filts = np.flipud(filts[1:filts.shape[0] - 1, :])
    else:  # odd -- don't take the DC
        neg_filts = np.flipud(filts[1:filts.shape[0], :])
    fft_filts = np.vstack((filts, neg_filts))
    # we need to switch representation to apply filters to fft of the signal, not sure why, but do it here
    return fft_filts.T


def make_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor,
                        padding_size=None, full_filter=True, strict=True,
                        bandwidth_scale_factor=1.0, include_lowpass=True,
                        include_highpass=True, filter_spacing='erb'):
    """Create cosine filters, oversampled by a factor provided by "sample_factor"
    Args:
        signal_length (int): Length of signal to be filtered with the generated
            filterbank. The signal length determines the length of the filters.
        sr (int): Sampling rate associated with the signal waveform.
        n (int): Number of filters (subbands) to be generated with standard
            sampling (i.e., using a sampling factor of 1). Note, the actual number of
            filters in the generated filterbank depends on the sampling factor, and
            may optionally include lowpass and highpass filters that allow for
            perfect reconstruction of the input signal (the exact number of lowpass
            and highpass filters is determined by the sampling factor). The
            number of filters in the generated filterbank is given below:
        +---------------+---------------+-+------------+---+---------------------+
        | sample factor |     n_out     |=|  bandpass  |\ +|  highpass + lowpass |
        +===============+===============+=+============+===+=====================+
        |      1        |     n+2       |=|     n      |\ +|     1     +    1    |
        +---------------+---------------+-+------------+---+---------------------+
        |      2        |   2*n+1+4     |=|   2*n+1    |\ +|     2     +    2    |
        +---------------+---------------+-+------------+---+---------------------+
        |      4        |   4*n+3+8     |=|   4*n+3    |\ +|     4     +    4    |
        +---------------+---------------+-+------------+---+---------------------+
        |      s        | s*(n+1)-1+2*s |=|  s*(n+1)-1 |\ +|     s     +    s    |
        +---------------+---------------+-+------------+---+---------------------+
        low_lim (int): Lower limit of frequency range. Filters will not be defined
            below this limit.
        hi_lim (int): Upper limit of frequency range. Filters will not be defined
            above this limit.
        sample_factor (int): Positive integer that determines how densely ERB function
            will be sampled to create bandpass filters. 1 represents standard sampling;
            adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
            adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
            adjacent bandpass filters will overlap by 87.5%.
        padding_size (int, optional): If None (default), the signal will not be padded
            before filtering. Otherwise, the filters will be created assuming the
            waveform signal will be padded to length padding_size*signal_length.
        full_filter (bool, default=True): If True (default), the complete filter that
            is ready to apply to the signal is returned. If False, only the first
            half of the filter is returned (likely positive terms of FFT).
        strict (bool, default=True): If True (default), will throw an error if
            sample_factor is not a power of two. This facilitates comparison across
            sample_factors. Also, if True, will throw an error if provided hi_lim
            is greater than the Nyquist rate.
        bandwidth_scale_factor (float, default=1.0): scales the bandpass filter bandwidths.
            bandwidth_scale_factor=2.0 means half-cosine filters will be twice as wide.
            Note that values < 1 will cause frequency gaps between the filters.
            bandwidth_scale_factor requires sample_factor=1, include_lowpass=False, include_highpass=False.
        include_lowpass (bool, default=True): if set to False, lowpass filter will be discarded.
        include_highpass (bool, default=True): if set to False, highpass filter will be discarded.
        filter_spacing (str, default='erb'): Specifies the type of reference spacing for the
            half-cosine filters. Options include 'erb' and 'linear'.
    Returns:
        tuple:
            A tuple containing the output:
            * **filts** (*array*)-- The filterbank consisting of filters have
                cosine-shaped frequency responses, with center frequencies equally
                spaced from low_lim to hi_lim on a scale specified by filter_spacing
            * **center_freqs** (*array*) -- center frequencies of filterbank in filts
            * **freqs** (*array*) -- freq vector in Hz, same frequency dimension as filts
    Raises:
        ValueError: Various value errors for bad choices of sample_factor or frequency
        limits; see description for strict parameter.
        UserWarning: Raises warning if cochlear filters exceed the Nyquist
        limit or go below 0.
        NotImplementedError: Raises error if specified filter_spacing is not implemented
    """

    # Specifiy the type of filter spacing, if using linear filters instead
    if filter_spacing == 'erb':
        _freq2ref = freq2erb
        _ref2freq = erb2freq
    elif filter_spacing == 'erb_r':
        _freq2ref = lambda x: freq2erb(hi_lim) - freq2erb(hi_lim - x)
        _ref2freq = lambda x: hi_lim - erb2freq(freq2erb(hi_lim) - x)
    elif (filter_spacing == 'lin') or (filter_spacing == 'linear'):
        _freq2ref = lambda x: x
        _ref2freq = lambda x: x
    elif 'random' in filter_spacing:
        _freq2ref, _ref2freq = get_freq_rand_conversions(
            np.linspace(low_lim, hi_lim, n),
            seed=int(filter_spacing.split('-')[1].replace('seed', '')),
            minval=freq2erb(low_lim),
            maxval=freq2erb(hi_lim))
    else:
        raise NotImplementedError('unrecognized spacing mode: %s' % filter_spacing)
    print('[make_cos_filters_nx] using filter_spacing=`{}`'.format(filter_spacing))

    if not bandwidth_scale_factor == 1.0:
        assert sample_factor == 1, "bandwidth_scale_factor only supports sample_factor=1"
        assert include_lowpass == False, "bandwidth_scale_factor only supports include_lowpass=False"
        assert include_highpass == False, "bandwidth_scale_factor only supports include_highpass=False"

    if not isinstance(sample_factor, int):
        raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
    if sample_factor <= 0:
        raise ValueError('sample_factor must be positive')

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = 'sample_factor odd, and will change filter widths. Use even sample factors for comparison.'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    # verify the high limit is allowed by the sampling rate
    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'
        if strict:
              raise ValueError(msg)
        else:
              warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # changing the sampling density without changing the filter locations
    # (and, thereby changing their widths) requires that a certain number of filters
    # be used.
    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = np.linspace(0, max_freq, n_freqs + 1)
    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))

    # cutoffs are evenly spaced on the scale specified by filter_spacing; for ERB scale,
    # interpolate linearly in erb space then convert back.
    # Also return the actual spacing used to generate the sequence (in case numpy does
    # something weird)
    center_freqs, step_spacing = np.linspace(_freq2ref(low_lim), _freq2ref(hi_lim), n_filters + 2, retstep=True)  # +2 for bin endpoints
    # we need to exclude the endpoints
    center_freqs = center_freqs[1:-1]

    freqs_ref = _freq2ref(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * bandwidth_scale_factor * step_spacing
        h = center_freqs[i] + sample_factor * bandwidth_scale_factor * step_spacing
        if _ref2freq(h) > sr/2:
            cf = _ref2freq(center_freqs[i])
            msg = "High ERB cutoff of filter with cf={:.2f}Hz exceeds {:.2f}Hz (Nyquist frequency)"
            warnings.warn(msg.format(cf, sr/2))
        if _ref2freq(l) < 0:
            cf = _ref2freq(center_freqs[i])
            msg = 'Low ERB cutoff of filter with cf={:.2f}Hz is not strictly positive'
            warnings.warn(msg.format(cf))
        # the first sample_factor # of rows in filts will be lowpass filters
        filts[(freqs_ref > l) & (freqs_ref < h), i_offset] = make_cosine_filter(freqs_ref, l, h, convert_to_erb=False)

    # add lowpass and highpass filters (there will be sample_factor number of each)
    for i in range(sample_factor):
        # account for the fact that the first sample_factor # of filts are lowpass
        i_offset = i + sample_factor
        lp_h_ind = max(np.where(freqs < _ref2freq(center_freqs[i]))[0]) # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[:lp_h_ind+1, i_offset], 2))

        hp_l_ind = min(np.where(freqs > _ref2freq(center_freqs[-1-i]))[0])  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1-i_offset], 2))

        filts[:lp_h_ind+1, i] = lp_filt
        filts[hp_l_ind:, -1-i] = hp_filt

    # get center freqs for lowpass and highpass filters
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * step_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * step_spacing
    center_freqs = np.concatenate((cfs_low, center_freqs, cfs_hi))

    # ensure that squared freq response adds to one
    filts = filts / np.sqrt(sample_factor)

    # convert center freqs from ERB numbers to Hz
    center_freqs = _ref2freq(center_freqs)

    # rectify
    center_freqs[center_freqs < 0] = 1

    # discard highpass and lowpass filters, if requested
    if include_lowpass == False:
        filts = filts[:, sample_factor:]
        center_freqs = center_freqs[sample_factor:]
    if include_highpass == False:
        filts = filts[:, :-sample_factor]
        center_freqs = center_freqs[:-sample_factor]

    # make the full filter by adding negative components
    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def tflog10(x):
    """Implements log base 10 in tensorflow """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


@tf.custom_gradient
def stable_power_compression_norm_grad(x):
    """With this power compression function, the gradients from the power compression are not applied via backprop, we just pass the previous gradient onwards"""
    e = tf.nn.relu(x) # add relu to x to avoid NaN in loss
    p = tf.pow(e,0.3)
    def grad(dy): #try to check for nans before we clip the gradients. (use tf.where)
        return dy
    return p, grad


@tf.custom_gradient
def stable_power_compression(x):
    """Clip the gradients for the power compression and remove nans. Clipped values are (-1,1), so any cochleagram value below ~0.2 will be clipped."""
    e = tf.nn.relu(x) # add relu to x to avoid NaN in loss
    p = tf.pow(e,0.3)
    def grad(dy): #try to check for nans before we clip the gradients. (use tf.where)
        g = 0.3 * pow(e,-0.7)
        is_nan_values = tf.is_nan(g)
        replace_nan_values = tf.ones(tf.shape(g), dtype=tf.float32)*1
        return dy * tf.where(is_nan_values,replace_nan_values,tf.clip_by_value(g, -1, 1))
    return p, grad


def cochleagram_graph(nets, SIGNAL_SIZE, SR, ENV_SR=200, LOW_LIM=20, HIGH_LIM=8000, N=40, SAMPLE_FACTOR=4, compression='none', WINDOW_SIZE=1001, debug=False, subbands_ifft=False, pycoch_downsamp=False, linear_max=796.87416837456942, input_node='input_signal', mean_subtract=False, rms_normalize=False, SMOOTH_ABS = False, return_subbands_only=False, include_all_keys=False, rectify_and_lowpass_subbands=False, pad_factor=None, return_coch_params=False, rFFT=False, linear_params=None, custom_filts=None, custom_compression_op=None, erb_filter_kwargs={}, reshape_kell2018=False, include_subbands_noise=False, subbands_noise_mean=0., subbands_noise_stddev=0., rate_level_kwargs={}, preprocess_kwargs={}):
    """
    Creates a tensorflow cochleagram graph using the pycochleagram erb filters to create the cochleagram with the tensorflow functions.
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. At a minumum, nets['input_signal'] (or equivilant) should be defined containing a placeholder (if just constructing cochleagrams) or a variable (if optimizing over the cochleagrams), and can have a batch size>1.
    SIGNAL_SIZE : int
        the length of the audio signal used for the cochleagram graph
    SR : int
        raw sampling rate in Hz for the audio.
    ENV_SR : int
        the sampling rate for the cochleagram after downsampling
    LOW_LIM : int
        Lower frequency limits for the filters.
    HIGH_LIM : int
        Higher frequency limits for the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    compression : string. see include_compression for compression options
        determine compression type to use in the cochleagram graph. If return_subbands is true, compress the rectified subbands
    WINDOW_SIZE : int
        the size of a window to use for the downsampling filter
    debug : boolean
        Adds more nodes to the graph for explicitly defining the real and imaginary parts of the signal when set to True (default False).
    subbands_ifft : boolean
        If true, adds the ifft of the subbands to nets
    input_node : string
        Name of the top level of nets, this is the input into the cochleagram graph. 
    mean_subtract : boolean
        If true, subtracts the mean of the waveform (explicitly removes the DC offset)
    rms_normalize : Boolean # ONLY USE WHEN GENERATING COCHLEAGRAMS
        If true, divides the input signal by its RMS value, such that the RMS value of the sound going into the cochleagram generation is equal to 1. This option should be false if inverting cochleagrams, as it can cause problems with the gradients
    linear_max : float
        If default value, use 796.87416837456942, which is the 5th percentile from the speech dataset when it is rms normalized to a value of 1. This value is only used if the compression is 'linearbelow1', 'linearbelow1sqrt', 'stable_point3'
    SMOOTH_ABS : Boolean
        If True, uses a smoother version of the absolute value for the hilbert transform sqrt(10^-3 + real(env) + imag(env))
    return_subbands_only : Boolean
        If True, returns the non-envelope extracted subbands before taking the hilbert envelope as the output node of the graph
    include_all_keys : Boolean 
        If True, returns all of the cochleagram and subbands processing keys in the dictionary
    rectify_and_lowpass_subbands : Boolean
        If True, rectifies and lowpasses the subbands before returning them (only works with return_subbands_only)
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    return_coch_params : Boolean
        If True, returns the cochleagram generation parameters in addition to nets
    rFFT : Boolean
        If True, builds the graph using rFFT and irFFT operations whenever possible
    linear_params : list of floats
        used for the linear compression operation, [m, b] where the output of the compression is y=mx+b. m and b can be vectors of shape [1,num_filts,1] to apply different values to each frequency channel. 
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain th full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    custom_compression_op : None or tensorflow partial function
        if specified as a function, applies the tensorflow function as a custom compression operation. Should take the input node and 'name' as the arguments
    erb_filter_kwargs : dictionary
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters
    reshape_kell2018 : boolean (False)
        if true, reshapes the output cochleagram to be 256x256 as used by kell2018
    include_subbands_noise : boolean (False)
        if include_subbands_noise and return_subbands_only are both true, white noise is added to subbands after compression (this feature is currently only accessible when return_subbands_only == True)
    subbands_noise_mean : float
        sets mean of subbands white noise if include_subbands_noise == True
    subbands_noise_stddev : float
        sets standard deviation of subbands white noise if include_subbands_noise == True
    rate_level_kwargs : dictionary
        contains keyword arguments for AN_rate_level_function (used if compression == 'rate_level')
    preprocess_kwargs : dictionary
        contains keyword arguments for preprocess_input function (used to randomize input dB SPL)
        
    Returns
    -------
    nets : dictionary
        a dictionary containing the parts of the cochleagram graph. Top node in this graph is nets['output_tfcoch_graph']
    COCH_PARAMS : dictionary (Optional)
        a dictionary containing all of the input parameters into the function
    """
    if return_coch_params: 
        COCH_PARAMS = locals()
        COCH_PARAMS.pop('nets')

    # run preprocessing operations on the input (ie rms normalization, convert to complex)
    nets = preprocess_input(nets, SIGNAL_SIZE, input_node, mean_subtract, rms_normalize, rFFT, **preprocess_kwargs)

    # fft of the input
    nets = fft_of_input(nets, pad_factor,debug, rFFT)

    # Make a wrapper for the compression function so it can be applied to the cochleagram and the subbands
    compression_function = functools.partial(include_compression, compression=compression, linear_max=linear_max, linear_params=linear_params, rate_level_kwargs=rate_level_kwargs, custom_compression_op=custom_compression_op)  

    # make cochlear filters and compute the cochlear subbands
    nets = extract_cochlear_subbands(nets, SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, pad_factor, debug, subbands_ifft, return_subbands_only, rectify_and_lowpass_subbands, rFFT, custom_filts, erb_filter_kwargs, include_all_keys, compression_function, include_subbands_noise, subbands_noise_mean, subbands_noise_stddev)
       
    # Build the rest of the graph for the downsampled cochleagram, if we are returning the cochleagram or if we want to build the whole graph anyway. 
    if (not return_subbands_only) or include_all_keys: 
        # hilbert transform on subband fft
        nets = hilbert_transform_from_fft(nets, SR, SIGNAL_SIZE, pad_factor, debug, rFFT)

        # absolute value of the envelopes (and expand to one channel)
        nets = abs_envelopes(nets, SMOOTH_ABS)

        # downsample and rectified nonlinearity
        nets = downsample_and_rectify(nets, SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp)

        # compress cochleagram 
        nets = compression_function(nets, input_node_name='cochleagram_no_compression', output_node_name='cochleagram')

        if reshape_kell2018:
            nets, output_node_name_coch = reshape_coch_kell_2018(nets)
        else: 
            output_node_name_coch = 'cochleagram'

    if return_subbands_only:
        nets['output_tfcoch_graph'] = nets['subbands_time_processed']
    else: 
        nets['output_tfcoch_graph'] = nets[output_node_name_coch]

    # return 
    if return_coch_params:
        return nets, COCH_PARAMS    
    else: 
        return nets


def preprocess_input(nets, SIGNAL_SIZE, input_node, mean_subtract, rms_normalize, rFFT,
                     set_dBSPL=False, dBSPL_range=[60., 60.]):
    """
    Does preprocessing on the input (rms and converting to complex number)
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. should already contain input_node
    input_node : string
        Name of the top level of nets, this is the input into the cochleagram graph.
    mean_subtract : boolean
        If true, subtracts the mean of the waveform (explicitly removes the DC offset)
    rms_normalize : Boolean # TODO: incorporate stable gradient code for RMS
        If true, divides the input signal by its RMS value, such that the RMS value of the sound going 
    rFFT : Boolean
        If true, preprocess input for using the rFFT operations
    set_dBSPL : Boolean
        If true, re-scale input waveform to dB SPL sampled uniformly from dBSPL_range
    dBSPL_range : list
        Range of sound presentation levels in units of dB re 20e-6 Pa ([minval, maxval])
    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph.
    """
    
    if rFFT:
        if SIGNAL_SIZE%2!=0:
            print('rFFT is only tested with even length signals. Change your input length.')
            return
    
    processed_input_node = input_node
    
    if mean_subtract:
        processed_input_node = processed_input_node + '_mean_subtract'
        nets[processed_input_node] = nets[input_node] - tf.reshape(tf.reduce_mean(nets[input_node],1),(-1,1))
        input_node = processed_input_node 
    
    if rms_normalize: # TODO: incoporate stable RMS normalization
        processed_input_node = processed_input_node + '_rms_normalized'
        nets['rms_input'] = tf.sqrt(tf.reduce_mean(tf.square(nets[input_node]), 1))
        nets[processed_input_node] = tf.identity(nets[input_node]/tf.reshape(nets['rms_input'],(-1,1)),'rms_normalized_input')
        input_node = processed_input_node
    
    if set_dBSPL: # NOTE: unstable if RMS of input is zero
        processed_input_node = processed_input_node + '_set_dBSPL'
        assert rms_normalize == False, "rms_normalize must be False if set_dBSPL=True"
        assert len(dBSPL_range) == 2, "dBSPL_range must be specified as [minval, maxval]"
        nets['dBSPL_set'] = tf.random.uniform([tf.shape(nets[input_node])[0], 1],
                                              minval=dBSPL_range[0], maxval=dBSPL_range[1],
                                              dtype=nets[input_node].dtype, name='sample_dBSPL_set')
        nets['rms_set'] = 20e-6 * tf.math.pow(10., nets['dBSPL_set'] / 20.)
        nets['rms_input'] = tf.sqrt(tf.reduce_mean(tf.square(nets[input_node]), axis=1, keepdims=True)) 
        nets[processed_input_node] = tf.math.multiply(nets['rms_set'] / nets['rms_input'], nets[input_node],
                                                      name='scale_input_to_dBSPL_set')
        input_node = processed_input_node
    
    if not rFFT:
        nets['input_signal_i'] = nets[input_node]*0.0
        nets['input_signal_complex'] = tf.complex(nets[input_node], nets['input_signal_i'], name='input_complex')
    else:
        nets['input_real'] = nets[input_node]
    return nets


def fft_of_input(nets, pad_factor, debug, rFFT):
    """
    Computs the fft of the signal and adds appropriate padding
    
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'subbands' are used for the hilbert transform
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    debug : boolean
        Adds more nodes to the graph for explicitly defining the real and imaginary parts of the signal when set to True.
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible
    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph with the rFFT of the input
    """
    # fft of the input
    if not rFFT:
        if pad_factor is not None:
            nets['input_signal_complex'] = tf.concat([nets['input_signal_complex'], tf.zeros([nets['input_signal_complex'].get_shape()[0], nets['input_signal_complex'].get_shape()[1]*(pad_factor-1)], dtype=tf.complex64)], axis=1)
        nets['fft_input'] = tf.fft(nets['input_signal_complex'],name='fft_of_input')
    else: 
        nets['fft_input'] = tf.spectral.rfft(nets['input_real'],name='fft_of_input') # Since the DFT of a real signal is Hermitian-symmetric, RFFT only returns the fft_length / 2 + 1 unique components of the FFT: the zero-frequency term, followed by the fft_length / 2 positive-frequency terms.

    nets['fft_input'] = tf.expand_dims(nets['fft_input'], 1, name='exd_fft_of_input')

    if debug: # return the real and imaginary parts of the fft separately
        nets['fft_input_r'] = tf.real(nets['fft_input'])
        nets['fft_input_i'] = tf.imag(nets['fft_input'])
    return nets


def extract_cochlear_subbands(nets, SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, pad_factor, debug, subbands_ifft, return_subbands_only, rectify_and_lowpass_subbands, rFFT, custom_filts, erb_filter_kwargs, include_all_keys, compression_function, include_subbands_noise, subbands_noise_mean, subbands_noise_stddev):
    """
    Computes the cochlear subbands from the fft of the input signal
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'fft_input' is multiplied by the cochlear filters
    SIGNAL_SIZE : int
        the length of the audio signal used for the cochleagram graph
    SR : int
        raw sampling rate in Hz for the audio.
    LOW_LIM : int
        Lower frequency limits for the filters.
    HIGH_LIM : int
        Higher frequency limits for the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    debug : boolean
        Adds more nodes to the graph for explicitly defining the real and imaginary parts of the signal
    subbands_ifft : boolean
        If true, adds the ifft of the subbands to nets
    return_subbands_only : Boolean
        If True, returns the non-envelope extracted subbands before taking the hilbert envelope as the output node of the graph
    rectify_and_lowpass_subbands : Boolean
        If True, rectifies and lowpasses the subbands before returning them (only works with return_subbands_only)
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain th full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    erb_filter_kwargs : dictionary
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters
    include_all_keys : Boolean
        If True, includes the time subbands and the cochleagram in the dictionary keys
    compression_function : function
        A partial function that takes in nets and the input and output names to apply compression 
    include_subbands_noise : boolean (False)
        if include_subbands_noise and return_subbands_only are both true, white noise is added to subbands after compression (this feature is currently only accessible when return_subbands_only == True)
    subbands_noise_mean : float
        sets mean of subbands white noise if include_subbands_noise == True
    subbands_noise_stddev : float
        sets standard deviation of subbands white noise if include_subbands_noise == True
    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph.
    """

    # make the erb filters tensor
    nets['filts_tensor'] = make_filts_tensor(SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, use_rFFT=rFFT, pad_factor=pad_factor, custom_filts=custom_filts, erb_filter_kwargs=erb_filter_kwargs)

    # make subbands by multiplying filts with fft of input
    nets['subbands'] = tf.multiply(nets['filts_tensor'],nets['fft_input'],name='mul_subbands')
    if debug: # return the real and imaginary parts of the subbands separately -- use if matching to their output
        nets['subbands_r'] = tf.real(nets['subbands'])
        nets['subbands_i'] = tf.imag(nets['subbands'])

    # TODO: with  using subbands_ifft is redundant. 
    # make the time subband operations if we are returning the subbands or if we want to include all of the keys in the graph
    if subbands_ifft or return_subbands_only or include_all_keys:
        if not rFFT:
            nets['subbands_ifft'] = tf.real(tf.ifft(nets['subbands'],name='ifft_subbands'),name='ifft_subbands_r')
        else:
            nets['subbands_ifft'] = tf.spectral.irfft(nets['subbands'],name='ifft_subbands')
        if return_subbands_only or include_all_keys:
            nets['subbands_time'] = nets['subbands_ifft']
            if rectify_and_lowpass_subbands: # TODO: the subband operations are hard coded in?
                nets['subbands_time_relu'] = tf.nn.relu(nets['subbands_time'], name='rectified_subbands')
                nets['subbands_time_lowpassed'] = hanning_pooling_1d_no_depthwise(nets['subbands_time_relu'], downsample=2, length_of_window=2*4, make_plots=False, data_format='NCW', normalize=True, sqrt_window=False)

    # TODO: noise is only added in the case when we are calcalculating the time subbands, but we might want something similar for the cochleagram
    if return_subbands_only or include_all_keys:
        # Compress subbands if specified and add noise. 
        nets = compression_function(nets, input_node_name='subbands_time_lowpassed', output_node_name='subbands_time_lowpassed_compressed')
        if include_subbands_noise:
            nets = add_neural_noise(nets, subbands_noise_mean, subbands_noise_stddev, input_node_name='subbands_time_lowpassed_compressed', output_node_name='subbands_time_lowpassed_compressed_with_noise')
            nets['subbands_time_lowpassed_compressed_with_noise'] = tf.expand_dims(nets['subbands_time_lowpassed_compressed_with_noise'],-1)
            nets['subbands_time_processed'] = nets['subbands_time_lowpassed_compressed_with_noise']
        else:
            nets['subbands_time_lowpassed_compressed'] = tf.expand_dims(nets['subbands_time_lowpassed_compressed'],-1)
            nets['subbands_time_processed'] = nets['subbands_time_lowpassed_compressed']

    return nets


def hilbert_transform_from_fft(nets, SR, SIGNAL_SIZE, pad_factor, debug, rFFT):
    """
    Performs the hilbert transform from the subband FFT -- gets ifft using only the real parts of the signal
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'subbands' are used for the hilbert transform
    SR : int
        raw sampling rate in Hz for the audio.
    SIGNAL_SIZE : int
        the length of the audio signal used for the cochleagram graph
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    debug : boolean
        Adds more nodes to the graph for explicitly defining the real and imaginary parts of the signal when set to True.
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible
    """

    if not rFFT:
        # make the step tensor for the hilbert transform (only keep the real components)
        if pad_factor is not None:
            freq_signal = np.fft.fftfreq(SIGNAL_SIZE*pad_factor, 1./SR)
        else:
            freq_signal = np.fft.fftfreq(SIGNAL_SIZE,1./SR)
        nets['step_tensor'] = make_step_tensor(freq_signal)

        # envelopes in frequency domain -- hilbert transform of the subbands
        nets['envelopes_freq'] = tf.multiply(nets['subbands'],nets['step_tensor'],name='env_freq')
    else:
        # make the padding to turn rFFT into a step function
        num_filts = nets['filts_tensor'].get_shape().as_list()[1]
#         num_batch = nets['subbands'].get_shape().as_list()[0]
        num_batch = tf.shape(nets['subbands'])[0]
        # TODO: this also might be a problem when we have pad_factor > 1
        print(num_batch)
        print(num_filts)
        print(int(SIGNAL_SIZE/2)-1)
        nets['hilbert_padding'] = tf.zeros([num_batch,num_filts,int(SIGNAL_SIZE/2)-1], tf.complex64) 
        nets['envelopes_freq'] = tf.concat([nets['subbands'],nets['hilbert_padding']],2,name='env_freq')

    if debug: # return real and imaginary parts separately
        nets['envelopes_freq_r'] = tf.real(nets['envelopes_freq'])
        nets['envelopes_freq_i'] = tf.imag(nets['envelopes_freq'])

    # fft of the envelopes.
    nets['envelopes_time'] = tf.ifft(nets['envelopes_freq'],name='ifft_envelopes')

    if not rFFT: # TODO: was this a bug in pycochleagram where the pad factor doesn't actually work? 
        if pad_factor is not None:
            nets['envelopes_time'] = nets['envelopes_time'][:,:,:SIGNAL_SIZE]

    if debug: # return real and imaginary parts separately
        nets['envelopes_time_r'] = tf.real(nets['envelopes_time'])
        nets['envelopes_time_i'] = tf.imag(nets['envelopes_time'])

    return nets


def abs_envelopes(nets, SMOOTH_ABS):
    """
    Absolute value of the envelopes (and expand to one channel), analytic hilbert signal
    
    Parameters
    ----------
    nets : dictionary
        dictionary containing the cochleagram graph. Downsampling will be applied to 'envelopes_time'
    SMOOTH_ABS : Boolean
        If True, uses a smoother version of the absolute value for the hilbert transform sqrt(10^-3 + real(env) + imag(env))
    Returns
    -------
    nets : dictionary
        dictionary containing the updated cochleagram graph
    """

    if SMOOTH_ABS:
        nets['envelopes_abs'] = tf.sqrt(1e-10 + tf.square(tf.real(nets['envelopes_time'])) + tf.square(tf.imag(nets['envelopes_time'])))
    else:
        nets['envelopes_abs'] = tf.abs(nets['envelopes_time'], name='complex_abs_envelopes')
    nets['envelopes_abs'] = tf.expand_dims(nets['envelopes_abs'],3, name='exd_abs_real_envelopes')
    return nets


def downsample_and_rectify(nets, SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp):
    """
    Downsamples the cochleagram and then performs rectification on the output (in case the downsampling results in small negative numbers)
    Parameters
    ----------
    nets : dictionary 
        dictionary containing the cochleagram graph. Downsampling will be applied to 'envelopes_abs'
    SR : int
        raw sampling rate of the audio signal
    ENV_SR : int
        end sampling rate of the envelopes
    WINDOW_SIZE : int
        the size of the downsampling window (should be large enough to go to zero on the edges).
    pycoch_downsamp : Boolean
        if true, uses a slightly different downsampling function
    Returns
    -------
    nets : dictionary
        dictionary containing parts of the cochleagram graph with added nodes for the downsampled subbands
    """
    # The stride for the downsample, works fine if it is an integer.
    DOWNSAMPLE = SR/ENV_SR
    if not ENV_SR == SR:
        # make the downsample tensor
        nets['downsample_filt_tensor'] = make_downsample_filt_tensor(SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp=pycoch_downsamp)
        nets['cochleagram_preRELU']  = tf.nn.conv2d(nets['envelopes_abs'], nets['downsample_filt_tensor'], [1, 1, DOWNSAMPLE, 1], 'SAME',name='conv2d_cochleagram_raw')
    else:
        nets['cochleagram_preRELU'] = nets['envelopes_abs']
    nets['cochleagram_no_compression'] = tf.nn.relu(nets['cochleagram_preRELU'], name='coch_no_compression')

    return nets


def include_compression(nets, compression='none', linear_max=796.87416837456942, input_node_name='cochleagram_no_compression', output_node_name='cochleagram', linear_params=None, rate_level_kwargs={}, custom_compression_op=None):
    """
    Choose compression operation to use and adds appropriate nodes to nets
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. Compression will be applied to input_node_name
    compression : string
        type of compression to perform
    linear_max : float
        used for the linearbelow compression operations (compression is linear below a value and compressed above it)
    input_node_name : string
        name in nets to apply the compression
    output_node_name : string
        name in nets that will be used for the following operation (default is cochleagram, but if returning subbands than it can be chaged)
    linear_params : list of floats
        used for the linear compression operation, [m, b] where the output of the compression is y=mx+b. m and b can be vectors of shape [1,num_filts,1] to apply different values to each frequency channel.
    custom_compression_op : None or tensorflow partial function
        if specified as a function, applies the tensorflow function as a custom compression operation. Should take the input node and 'name' as the arguments
    Returns
    -------
    nets : dictionary
        dictionary containing parts of the cochleagram graph with added nodes for the compressed cochleagram 
    """
    # compression of the cochleagram
    if compression=='quarter':
        nets[output_node_name] = tf.sqrt(tf.sqrt(nets[input_node_name], name=output_node_name))
    elif compression=='quarter_plus':
        nets[output_node_name] = tf.sqrt(tf.sqrt(nets[input_node_name]+1e-01, name=output_node_name))
    elif compression=='point3':
        nets[output_node_name] = tf.pow(nets[input_node_name],0.3, name=output_node_name)
    elif compression=='stable_point3':
        nets[output_node_name] = tf.identity(stable_power_compression(nets[input_node_name]*linear_max),name=output_node_name) 
    elif compression=='stable_point3_norm_grads':
        nets[output_node_name] = tf.identity(stable_power_compression_norm_grad(nets[input_node_name]*linear_max),name=output_node_name) 
    elif compression=='linearbelow1':
        nets[output_node_name] = tf.where((nets[input_node_name]*linear_max)<1, nets[input_node_name]*linear_max, tf.pow(nets[input_node_name]*linear_max,0.3), name=output_node_name)
    elif compression=='stable_linearbelow1':
        nets['stable_power_compressed_%s'%output_node_name] = tf.identity(stable_power_compression(nets[input_node_name]*linear_max),name='stable_power_compressed_%s'%output_node_name)
        nets[output_node_name] = tf.where((nets[input_node_name]*linear_max)<1, nets[input_node_name]*linear_max, nets['stable_power_compressed_%s'%output_node_name], name=output_node_name)
    elif compression=='linearbelow1sqrt':
        nets[output_node_name] = tf.where((nets[input_node_name]*linear_max)<1, nets[input_node_name]*linear_max, tf.sqrt(nets[input_node_name]*linear_max), name=output_node_name)
    elif compression=='quarter_clipped':
        nets[output_node_name] = tf.sqrt(tf.sqrt(tf.maximum(nets[input_node_name],1e-01), name=output_node_name))
    elif compression=='none':
        nets[output_node_name] = nets[input_node_name]
    elif compression=='sqrt':
        nets[output_node_name] = tf.sqrt(nets[input_node_name], name=output_node_name)
    elif compression=='dB': # NOTE: this compression does not work well for the backwards pass, results in nans
        nets[output_node_name + '_noclipped'] = 20 * tflog10(nets[input_node_name])/tf.reduce_max(nets[input_node_name])
        nets[output_node_name] = tf.maximum(nets[output_node_name + '_noclipped'], -60)
    elif compression=='dB_plus': # NOTE: this compression does not work well for the backwards pass, results in nans
        nets[output_node_name + '_noclipped'] = 20 * tflog10(nets[input_node_name]+1)/tf.reduce_max(nets[input_node_name]+1)
        nets[output_node_name] = tf.maximum(nets[output_node_name + '_noclipped'], -60, name=output_node_name)
    elif compression=='linear':
        assert (type(linear_params)==list) and len(linear_params)==2, "Specifying linear compression but not specifying the compression parameters in linear_params=[m, b]"
        nets[output_node_name] = linear_params[0]*nets[input_node_name] + linear_params[1]
    elif compression=='rate_level':
        nets[output_node_name] = AN_rate_level_function(nets[input_node_name], name=output_node_name, **rate_level_kwargs)
    elif compression=='custom':
        nets[output_node_name] = custom_compression_op(nets[input_node_name], name=output_node_name)

    return nets


def make_step_tensor(freq_signal):
    """
    Make step tensor for calcaulting the anlyatic envelopes.
    Parameters
    __________
    freq_signal : array
        numpy array containing the frequenies of the audio signal (as calculated by np.fft.fftfreqs).
    Returns
    -------
    step_tensor : tensorflow tensor
        tensorflow tensor with dimensions [0 len(freq_signal) 0 0] as a step function where frequencies > 0 are 1 and frequencies < 0 are 0.
    """
    step_func = (freq_signal>=0).astype(np.int)*2 # wikipedia says that this should be 2x the original.
    step_func[freq_signal==0] = 0 # https://en.wikipedia.org/wiki/Analytic_signal (this shouldn't actually matter i think.
    step_tensor = tf.constant(step_func, dtype=tf.complex64)
    step_tensor = tf.expand_dims(step_tensor, 0)
    step_tensor = tf.expand_dims(step_tensor, 1)
    return step_tensor


def make_filts_tensor(SIGNAL_SIZE, SR=16000, LOW_LIM=20, HIGH_LIM=8000, N=40, SAMPLE_FACTOR=4, use_rFFT=False, pad_factor=None, custom_filts=None, erb_filter_kwargs={}):
    """
    Use pycochleagram to make the filters using the specified prameters (make_erb_cos_filters_nx). Then input them into a tensorflow tensor to be used in the tensorflow cochleagram graph.
    Parameters
    ----------
    SIGNAL_SIZE: int
        length of the audio signal to convert, and the size of cochleagram filters to make.
    SR : int
        raw sampling rate in Hz for the audio.
    LOW_LIM : int
        Lower frequency limits for the filters.
    HIGH_LIM : int
        Higher frequency limits for the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    use_rFFT : Boolean
        if True, the only returns the first half of the filters, corresponding to the positive component. 
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain th full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    erb_filter_kwargs : dictionary 
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters
    Returns
    -------
    filts_tensor : tensorflow tensor, complex
        tensorflow tensor with dimensions [0 SIGNAL_SIZE NUMBER_OF_FILTERS] that includes the erb filters created from make_erb_cos_filters_nx in pycochleagram
    """
    if pad_factor:
        padding_size = (pad_factor-1)*SIGNAL_SIZE
    else:
        padding_size=None

    if custom_filts is None: 
        # make the filters
        filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(SIGNAL_SIZE, SR, N, LOW_LIM, HIGH_LIM, SAMPLE_FACTOR, padding_size=padding_size, **erb_filter_kwargs) #TODO: decide if we want to change the pad_factor and full_filter arguments.
    else: # TODO: ADD CHECKS TO MAKE SURE THAT THESE MATCH UP WITH THE INPUT SIGNAL 
        assert custom_filts.shape[1] == SIGNAL_SIZE, "CUSTOM FILTER SHAPE DOES NOT MATCH THE INPUT AUDIO SHAPE"
        filts = custom_filts

    if not use_rFFT: 
        filts_tensor = tf.constant(filts, tf.complex64)
    else: # TODO I believe that this is where the padd factor problem comes in! We are only using part of the signal here. 
        filts_tensor = tf.constant(filts[:,0:(int(SIGNAL_SIZE/2)+1)], tf.complex64)

    filts_tensor = tf.expand_dims(filts_tensor, 0)

    return filts_tensor


def make_downsample_filt_tensor(SR=16000, ENV_SR=200, WINDOW_SIZE=1001, pycoch_downsamp=False):
    """
    Make the sinc filter that will be used to downsample the cochleagram
    Parameters
    ----------
    SR : int
        raw sampling rate of the audio signal
    ENV_SR : int
        end sampling rate of the envelopes
    WINDOW_SIZE : int
        the size of the downsampling window (should be large enough to go to zero on the edges).
    pycoch_downsamp : Boolean
        if true, uses a slightly different downsampling function
    Returns
    -------
    downsample_filt_tensor : tensorflow tensor, tf.float32
        a tensor of shape [0 WINDOW_SIZE 0 0] the sinc windows with a kaiser lowpass filter that is applied while downsampling the cochleagram
    """
    DOWNSAMPLE = SR/ENV_SR
    if not pycoch_downsamp: 
        downsample_filter_times = np.arange(-WINDOW_SIZE/2,int(WINDOW_SIZE/2))
        downsample_filter_response_orig = np.sinc(downsample_filter_times/DOWNSAMPLE)/DOWNSAMPLE
        downsample_filter_window = signal.kaiser(WINDOW_SIZE, 5)
        downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    else: 
        max_rate = DOWNSAMPLE
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
        if max_rate!=1:    
            downsample_filter_response = signal.firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))
        else:  # just in case we aren't downsampling -- I think this should work? 
            downsample_filter_response = zeros(2 * half_len + 1)
            downsample_filter_response[half_len + 1] = 1
            
        # Zero-pad our filter to put the output samples at the center
        # n_pre_pad = int((DOWNSAMPLE - half_len % DOWNSAMPLE))
        # n_post_pad = 0
        # n_pre_remove = (half_len + n_pre_pad) // DOWNSAMPLE
        # We should rarely need to do this given our filter lengths...
        # while _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis],
        #                  up, down) < n_out + n_pre_remove:
        #     n_post_pad += 1
        # downsample_filter_response = np.concatenate((np.zeros(n_pre_pad), downsample_filter_response, np.zeros(n_post_pad)))
            
    downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)

    return downsample_filt_tensor


def add_neural_noise(nets, subbands_noise_mean, subbands_noise_stddev, input_node_name='subbands_time_lowpassed_compressed', output_node_name='subbands_time_lowpassed_compressed_with_noise'):
    # Add white noise variable with the same size to the rectified and compressed subbands
    nets['neural_noise'] = tf.random.normal(tf.shape(nets[input_node_name]), mean=subbands_noise_mean,
                                            stddev=subbands_noise_stddev, dtype=nets[input_node_name].dtype)
    nets[output_node_name] = tf.nn.relu(tf.math.add(nets[input_node_name], nets['neural_noise']))
    return nets

                       
def reshape_coch_kell_2018(nets):
    """
    Wrapper to reshape the cochleagram to 256x256 similar to that used in kell2018.
    Note that this function relies on tf.image.resize_images which can have unexpected behavior... use with caution.
    nets : dictionary
        dictionary containing parts of the cochleagram graph. should already contain cochleagram
    """
    print('### WARNING: tf.image.resize_images is not trusted, use caution ###')
    nets['min_cochleagram'] = tf.reduce_min(nets['cochleagram'])
    nets['max_cochleagram'] = tf.reduce_max(nets['cochleagram'])
    # it is possible that this scaling is going to mess up the gradients for the waveform generation
    nets['scaled_cochleagram'] = 255*(1-((nets['max_cochleagram']-nets['cochleagram'])/(nets['max_cochleagram']-nets['min_cochleagram'])))
    nets['reshaped_cochleagram'] = tf.image.resize_images(nets['scaled_cochleagram'],[256,256], align_corners=False, preserve_aspect_ratio=False)
    return nets, 'reshaped_cochleagram'


def convert_Pa_to_dBSPL(pa):
    """ Converts units of Pa to dB re 20e-6 Pa (dB SPL) """
    return 20. * np.log10(pa / 20e-6)


def convert_dBSPL_to_Pa(dbspl):
    """ Converts units of dB re 20e-6 Pa (dB SPL) to Pa """
    return 20e-6 * np.power(10., dbspl / 20.)


def AN_rate_level_function(tensor_subbands, name='rate_level_fcn', rate_spont=70., rate_max=250.,
                           rate_normalize=True, beta=3., halfmax_dBSPL=20.):
    """
    Function implements the auditory nerve rate-level function described by Peter Heil
    and colleagues (2011, J. Neurosci.): the "amplitude-additivity model".
    
    Args
    ----
    tensor_subbands (tensor): shape must be [batch, freq, time, (channel)], units are Pa
    name (str): name for the tensorflow operation
    rate_spont (float): spontaneous spiking rate (spikes/s)
    rate_max (float): maximum spiking rate (spikes/s)
    rate_normalize (bool): if True, output will be re-scaled between 0 and 1
    beta (float or list): determines the steepness of rate-level function (dimensionless)
    halfmax_dBSPL (float or list): determines threshold of rate-level function (units dB SPL)
    
    Returns
    -------
    tensor_rates (tensor): same shape as tensor_subbands, units are spikes/s or normalized
    """
    # Check arguments and compute shape for frequency-channel-specific parameters
    assert rate_spont > 0, "rate_spont must be greater than zero to avoid division by zero"
    if len(tensor_subbands.shape) == 3:
        freq_specific_shape = [tensor_subbands.shape[1], 1]
    elif len(tensor_subbands.shape) == 4:
        freq_specific_shape = [tensor_subbands.shape[1], 1, 1]
    else:
        raise ValueError("tensor_subbands must have shape [batch, freq, time, (channel)]")
    # Convert beta to tensor (can be a single value or frequency channel specific)
    beta = np.array(beta).reshape([-1])
    assert_msg = "beta must be one value or a list of length {}".format(tensor_subbands.shape[1])
    assert len(beta) == 1 or len(beta) == tensor_subbands.shape[1], assert_msg
    beta_vals = tf.constant(beta,
                            dtype=tensor_subbands.dtype,
                            shape=freq_specific_shape)
    # Convert halfmax_dBSPL to tensor (can be a single value or frequency channel specific)
    halfmax_dBSPL = np.array(halfmax_dBSPL).reshape([-1])
    assert_msg = "halfmax_dBSPL must be one value or a list of length {}".format(tensor_subbands.shape[1])
    assert len(halfmax_dBSPL) == 1 or len(halfmax_dBSPL) == tensor_subbands.shape[1], assert_msg
    P_halfmax = tf.constant(convert_dBSPL_to_Pa(halfmax_dBSPL),
                            dtype=tensor_subbands.dtype,
                            shape=freq_specific_shape)
    # Convert rate_spont and rate_max to tf.constants (single values)
    R_spont = tf.constant(rate_spont, dtype=tensor_subbands.dtype, shape=[])
    R_max = tf.constant(rate_max, dtype=tensor_subbands.dtype, shape=[])
    # Implementation analogous to equation (8) from Heil et al. (2011, J. Neurosci.)
    P_0 = P_halfmax / (tf.pow((R_max + R_spont) / R_spont, 1/beta_vals) - 1)
    R_func = lambda P: R_max / (1 + ((R_max - R_spont) / R_spont) * tf.pow(P / P_0 + 1, -beta_vals))
    tensor_rates = tf.map_fn(R_func, tensor_subbands, name=name)
    # If rate_normalize is True, re-scale spiking rates to fall between 0 and 1
    if rate_normalize:
        tensor_rates = (tensor_rates - R_spont) / (R_max - R_spont)
    return tensor_rates


def make_hanning_kernel_1d(downsample=2, length_of_window=8, make_plots=False, normalize=False, sqrt_window=True):
    """
    Make the symmetric 1d hanning kernel to use for the pooling filters
    For downsample=2, using length_of_window=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For downsample=3, using length_of_window=12 gives a reduction of -28.607805482176282 at 1/6 cycles
    For downsample=4, using length_of_window=15 gives a reduction of -23 at 1/8 cycles
    We want to reduce the frequencies above the nyquist by at least 20dB.
    Parameters
    ----------
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version) -- normal window generation has sqrt_window=False
    Returns
    -------
    one_dimensional_kernel : numpy array
        hanning kernel in 1d to use as a kernel for filtering
    """

    window = 0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(length_of_window)) / (length_of_window - 1)))
    if sqrt_window:
        one_dimensional_kernel = np.sqrt(window)
    else:
        one_dimensional_kernel = window

    if normalize:
        one_dimensional_kernel = one_dimensional_kernel/sum(one_dimensional_kernel)
        window = one_dimensional_kernel

    if make_plots:
        A = np.fft.fft(window, 2048) / (len(window) / 2.0)
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20.0 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))

        nyquist = 1 / (2 * downsample)
        ny_idx = np.where(np.abs(freq - nyquist) == np.abs(freq - nyquist).min())[0][0]
        print(['Frequency response at ' + 'nyquist (%.3f Hz)'%nyquist + ' is ' + '%d'%response[ny_idx]])
        plt.figure()
        plt.plot(window)
        plt.title(r"Hanning window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.figure()
        plt.plot(freq, response)
        plt.axis([-0.5, 0.5, -120, 0])
        plt.title(r"Frequency response of the Hanning window")
        plt.ylabel("Normalized magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")

    return one_dimensional_kernel


def make_hanning_kernel_tensor_1d(n_channels, downsample=2, length_of_window=8, make_plots=False, normalize=False, sqrt_window=True):
    """
    Make a tensor containing the symmetric 1d hanning kernel to use for the pooling filters
    For downsample=2, using length_of_window=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For downsample=3, using length_of_window=12 gives a reduction of -28.607805482176282 at 1/6 cycles
    For downsample=4, using length_of_window=15 gives a reduction of -23 at 1/8 cycles
    We want to reduce the frequencies above the nyquist by at least 20dB.
    Parameters
    ----------
    n_channels : int
        number of channels to copy the kernel into
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version) -- normal window generation has sqrt_window=False
    Returns
    -------
    hanning_tensor : tensorflow tensor
        tensorflow tensor containing the hanning tensor with size [1 length_of_window n_channels 1]
    """
    hanning_kernel = make_hanning_kernel_1d(downsample=downsample,length_of_window=length_of_window,make_plots=make_plots, normalize=normalize, sqrt_window=sqrt_window)
    hanning_kernel = np.expand_dims(np.dstack([hanning_kernel.astype(np.float32)]*n_channels),axis=3)
    hanning_tensor = tf.constant(hanning_kernel)
    return hanning_tensor


def hanning_pooling_1d(input_tensor, downsample=2, length_of_window=8, make_plots=False, data_format='NWC', normalize=False, sqrt_window=True):
    """
    Parameters
    ----------
    input_tensor : tensorflow tensor
        tensor on which we will apply the hanning pooling operation
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters
    data_format : 'NWC' or 'NCW'
        Defaults to "NWC", the data is stored in the order of [batch, in_width, in_channels].
        The "NCW" format stores data as [batch, in_channels, in_width].
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version) -- normal window generation has sqrt_window=False
    Returns
    -------
    output_tensor : tensorflow tensor
        tensorflow tensor containing the downsampled input_tensor of shape corresponding to data_format
    """

    if data_format=='NWC':
        n_channels = input_tensor.get_shape().as_list()[2]
    elif data_format=='NCW':
        batch_size, n_channels, in_width = input_tensor.get_shape().as_list()
        input_tensor = tf.transpose(input_tensor, [0, 2, 1]) # reshape to [batch_size, in_wdith, in_channels]

    input_tensor = tf.expand_dims(input_tensor,1) # reshape to [batch_size, 1, in_width, in_channels]
    h_tensor = make_hanning_kernel_tensor_1d(n_channels, downsample=downsample, length_of_window=length_of_window, make_plots=make_plots, normalize=normalize, sqrt_window=sqrt_window)

    output_tensor = tf.nn.depthwise_conv2d(input_tensor, h_tensor, strides=[1, downsample, downsample, 1], padding='SAME', name='hpooling')

    output_tensor = tf.squeeze(output_tensor, name='squeeze_output')
    if data_format=='NWC':
        return output_tensor
    elif data_format=='NCW':
        return tf.transpose(output_tensor, [0, 2, 1]) # reshape to [batch_size, in_channels, out_width]


def make_hanning_kernel_tensor_1d_no_depthwise(n_channels, downsample=2, length_of_window=8, make_plots=False, normalize=False, sqrt_window=True):
    """
    Make a tensor containing the symmetric 1d hanning kernel to use for the pooling filters
    For downsample=2, using length_of_window=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For downsample=3, using length_of_window=12 gives a reduction of -28.607805482176282 at 1/6 cycles
    For downsample=4, using length_of_window=15 gives a reduction of -23 at 1/8 cycles
    We want to reduce the frequencies above the nyquist by at least 20dB.
    Parameters
    ----------
    n_channels : int
        number of channels to copy the kernel into
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version) -- normal window generation has sqrt_window=False
    Returns
    -------
    hanning_tensor : tensorflow tensor
        tensorflow tensor containing the hanning tensor with size [length_of_window, num_channels, num_channels]
    """
    hanning_kernel = make_hanning_kernel_1d(downsample=downsample,length_of_window=length_of_window,make_plots=make_plots, normalize=normalize, sqrt_window=sqrt_window).astype(np.float32)
    hanning_kernel_expanded = np.expand_dims(hanning_kernel,0) * np.expand_dims(np.eye(n_channels),3).astype(np.float32) # [n_channels, n_channels, filter_width]
    hanning_tensor = tf.constant(hanning_kernel_expanded) # [length_of_window, num_channels, num_channels]
    hanning_tensor = tf.transpose(hanning_tensor, [2, 0, 1])
    return hanning_tensor


def hanning_pooling_1d_no_depthwise(input_tensor, downsample=2, length_of_window=8, make_plots=False, data_format='NWC', normalize=False, sqrt_window=True):
    """
    Parameters
    ----------
    input_tensor : tensorflow tensor
        tensor on which we will apply the hanning pooling operation
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters
    data_format : 'NWC' or 'NCW'
        Defaults to "NWC", the data is stored in the order of [batch, in_width, in_channels].
        The "NCW" format stores data as [batch, in_channels, in_width].
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
make_hanning_kernel_tensor_1d_no_depthwise
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version) -- normal window generation has sqrt_window=False
    Returns
    -------
    output_tensor : tensorflow tensor
        tensorflow tensor containing the downsampled input_tensor of shape corresponding to data_format
    """

    if data_format=='NWC':
        n_channels = input_tensor.get_shape().as_list()[2]
    elif data_format=='NCW':
        batch_size, n_channels, in_width = input_tensor.get_shape().as_list()
        input_tensor = tf.transpose(input_tensor, [0, 2, 1]) # reshape to [batch_size, in_wdith, in_channels]

    h_tensor = make_hanning_kernel_tensor_1d_no_depthwise(n_channels, downsample=downsample, length_of_window=length_of_window, make_plots=make_plots, normalize=normalize, sqrt_window=sqrt_window)

    output_tensor = tf.nn.conv1d(input_tensor, h_tensor, stride=downsample, padding='SAME', name='hpooling')

    if data_format=='NWC':
        return output_tensor
    elif data_format=='NCW':
        return tf.transpose(output_tensor, [0, 2, 1]) # reshape to [batch_size, in_channels, out_width]


def build_cochlear_model(tensor_waveform,
                         signal_rate=20000,
                         filter_type='half-cosine',
                         filter_spacing='erb',
                         HIGH_LIM=8000,
                         LOW_LIM=20,
                         N=40,
                         SAMPLE_FACTOR=1,
                         bandwidth_scale_factor=1.0,
                         compression='stable_point3',
                         include_highpass=False,
                         include_lowpass=False,
                         linear_max=1.0,
                         rFFT=True,
                         rectify_and_lowpass_subbands=True,
                         return_subbands_only=True,
                         **kwargs):
    """
    This function serves as a wrapper for `tfcochleagram_graph` and builds the cochlear model graph.
    * * * * * * Default arguments are set to those used to train recognition networks * * * * * *
    
    Parameters
    ----------
    tensor_waveform (tensor): input signal waveform (with shape [batch, time])
    signal_rate (int): sampling rate of signal waveform in Hz
    filter_type (str): type of cochlear filters to build ('half-cosine')
    filter_spacing (str, default='erb'): Specifies the type of reference spacing for the
        half-cosine filters. Options include 'erb' and 'linear'.
    HIGH_LIM (float): high frequency cutoff of filterbank (only used for 'half-cosine')
    LOW_LIM (float): low frequency cutoff of filterbank (only used for 'half-cosine')
    N (int): number of cochlear bandpass filters
    SAMPLE_FACTOR (int): specifies how densely to sample cochlea (only used for 'half-cosine')
    bandwidth_scale_factor (float): factor by which to symmetrically scale the filter bandwidths
        bandwidth_scale_factor=2.0 means filters will be twice as wide.
        Note that values < 1 will cause frequency gaps between the filters.
    include_highpass (bool): determines if filterbank includes highpass filter(s) (only used for 'half-cosine')
    include_lowpass (bool): determines if filterbank includes lowpass filter(s) (only used for 'half-cosine')
    linear_max (float): used for the linearbelow compression operations
        (compression is linear below a value and compressed above it)
    rFFT (bool): If True, builds the graph using rFFT and irFFT operations whenever possible
    rectify_and_lowpass_subbands (bool): If True, rectifies and lowpass-filters subbands before returning
    return_subbands_only (bool): If True, returns subbands before taking the hilbert envelope as the output node
    kwargs (dict): additional keyword arguments passed directly to tfcochleagram_graph
    
    Returns
    -------
    tensor_cochlear_representation (tensor): output cochlear representation
    coch_container (dict): dictionary containing cochlear model stages
    """
    signal_length = tensor_waveform.get_shape().as_list()[-1]
    
    if filter_type == 'half-cosine':
        assert HIGH_LIM <= signal_rate/2, "cochlear filterbank high_lim is above Nyquist frequency"
        filts, center_freqs, freqs = make_cos_filters_nx(
            signal_length,
            signal_rate,
            N,
            LOW_LIM,
            HIGH_LIM,
            SAMPLE_FACTOR,
            padding_size=None,
            full_filter=True,
            strict=True,
            bandwidth_scale_factor=bandwidth_scale_factor,
            include_lowpass=include_lowpass,
            include_highpass=include_highpass,
            filter_spacing=filter_spacing)
        assert filts.shape[1] == signal_length, "filter array shape must match signal length"
    else:
        raise ValueError('Specified filter_type {} is not supported'.format(filter_type))
    
    coch_container = {'input_signal': tensor_waveform}
    coch_container = cochleagram_graph(
        coch_container,
        signal_length,
        signal_rate,
        LOW_LIM=LOW_LIM,
        HIGH_LIM=HIGH_LIM,
        N=N,
        SAMPLE_FACTOR=SAMPLE_FACTOR,
        custom_filts=filts,
        linear_max=linear_max,
        rFFT=rFFT,
        rectify_and_lowpass_subbands=rectify_and_lowpass_subbands,
        return_subbands_only=return_subbands_only,
        **kwargs)
    
    tensor_cochlear_representation = coch_container['output_tfcoch_graph']
    return tensor_cochlear_representation, coch_container
