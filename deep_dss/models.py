from deep_dss.helpers import set_constants, set_paths
from deepsphere import utils, models
import tensorflow as tf
import numpy as np

(NSIDE, NPIX, PIXEL_AREA, ORDER, BIAS, DENSITY_M, DENSITY_KG, ELLIP_SIGMA) = set_constants()

(PATH_TO_INPUT, PATH_TO_OUTPUT, PATH_TO_CHECKPOINTS, PATH_TO_VAL) = set_paths()


# Model parameters

def params_v2(verbose=True, path_to_checkpoints=""):
    """
    Returns params dict for v2 ("winning") architecture
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = path_to_checkpoints + "flask101-101-v2"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 64, 1]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5] * 5  # Polynomial orders.
    params['batch_norm'] = [True, True, True, True, False]  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 16]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of
    # weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 20  # Number of passes through the training data.
    params[
        'batch_size'] = 16 * ORDER ** 2  # Constant quantity of information (#pixels) per step (invariant to sample
    # size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(1.99e-4, step, decay_steps=1, decay_rate=0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'l1'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = 15

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v8(verbose=True, path_to_checkpoints=""):
    """
    Returns params dict for v8 architecture
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = path_to_checkpoints + "flask101-101-v8"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [32, 32, 64, 64, 64, 64, 32, 32]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [False] * 8  # Batch normalization.
    params['M'] = [2]  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 256]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 20  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: 1e-4
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'custom2'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = 60

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v9(verbose=True, path_to_checkpoints=""):
    """
    Returns params dict for v9 architecture
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = path_to_checkpoints + "flask101-101-v9"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [32, 32, 64, 64, 64, 64, 32, 32]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [False] * 8  # Batch normalization.
    params['M'] = [2]  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 256]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0.025  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 20  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: 1e-4
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'custom2'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = 60

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v10(verbose=True, path_to_checkpoints=""):
    """
    Returns params dict for v10 architecture
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = path_to_checkpoints + "flask101-101-v10"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [32, 32, 64, 64, 64, 64, 32, 32]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [False] * 8  # Batch normalization.
    params['M'] = [2]  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 256]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0.025  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 20  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: 1e-4
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'custom2'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = 60

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v11(verbose=True, num_epochs=20, learning_rate=1e-4, eval_frequency=3):
    """
    Returns params dict for v11 architecture
    :param eval_frequency: Evaluation frequency (# of batches)
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask101-101-v11"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [32, 32, 32, 32, 32, 32, 32, 1]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [True] * 8  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 128]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = num_epochs  # Number of passes through the training data.
    params['batch_size'] = 64  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: learning_rate
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'l1'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = eval_frequency

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v12(verbose=True, num_epochs=20, learning_rate=1e-4, eval_frequency=3):
    """
    Returns params dict for v12 architecture
    :param eval_frequency: Evaluation frequency (# of batches)
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask101-101-v12"

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'meanvar'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [32, 32, 32, 32, 32, 32, 32, 1]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [True] * 8  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.
    params['input_channel'] = 1  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 128]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, ORDER)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = num_epochs  # Number of passes through the training data.
    params['batch_size'] = 64  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: learning_rate
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'custom2'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = eval_frequency

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_vdata1(exp_name, input_channels, nmaps, nfilters, verbose=True, num_epochs=20, learning_rate=1e-4,
                  decay_factor=0.999,
                  order=ORDER, batch_size=16):
    """
    Returns params dict for vdata1 type architectures
    :param batch_size:
    :param nfilters:
    :param exp_name:
    :param order:
    :param decay_factor:
    :param nmaps:
    :param input_channels:
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flaskv2-vdata1-{}".format(exp_name)

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = None  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [nfilters] * 8  # Graph convolutional layers: number of feature maps.
    params['K'] = [5, 5, 5, 5, 5, 5, 5, 5]  # Polynomial orders.
    params['batch_norm'] = [True] * 7 + [False]  # Batch normalization.
    params['M'] = [1]  # Fully connected layers: output dimensionalities.
    params['input_channel'] = input_channels  # Two channels (spherical maps) per sample.

    # Pooling.
    nsides = [NSIDE, NSIDE // 2, NSIDE // 4, NSIDE // 8, NSIDE // 16, NSIDE // 32, NSIDE // 64, NSIDE // 128,
              NSIDE // 256]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, order)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = 0  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = 0.8  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = num_epochs  # Number of passes through the training data.
    params['batch_size'] = batch_size  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(learning_rate, step, decay_steps=1,
                                                                  decay_rate=decay_factor)
    # params['scheduler'] = lambda step: learning_rate
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = 'l1'  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = (12 * order * order * nmaps / batch_size) / 3  # Thrice per epoch

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // order) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // order) ** 2))
        print('=> #pixels for training (input): {:,}'.format(
            params['num_epochs'] * 12 * order * order * nmaps * (NSIDE // order) ** 2))

    return params


def params_v2_biasless(exp_name, convtype='chebyshev5', pooltype='max', nmaps=16,
                        activation_func='relu', stat_layer=None, input_channels=1,
                        gc_depth=8, nfilters=64,
                        const_k=5, var_k=None,
                        filters=None, batch_norm_output=False,
                        var_batch_norm=None,
                        fc_layers=[], num_outputs=1,
                        reg_factor=0, dropout_rate=0.8,
                        verbose=True, num_epochs=1,
                        learning_rate=1e-4,
                        decay_factor=0.999, decay_freq=1,
                        decay_staircase=False, loss_func="l1",
                        nside=NSIDE, nsides=None,
                        order=ORDER, batch_size=16):
    """
    Returns params dict for biasless 1 architectures

    :param convtype: Type of graph convolution performed ("chebyshev5" or "monomials").
    :param num_outputs: 1 for just sigma_8, 2 for sigma_8 and predicted log-variance q
    :param nsides: List of NSIDES for graph convolutional layers. Length = gc_depth
    :param nside: NSIDE of input maps. Should be 1024.
    :param loss_func: Choice of loss function ("l1", "custom1", "custom2", "l2"). Must be implemented in DeepSphere codebase.
    :param decay_staircase: If true, performs integer division in lr decay, decaying every decay_freq steps.
    :param decay_freq: If decay_staircase=true, acts to stagger decays. Otherwise, brings down decay factor.
    :param dropout_rate: Percentage of neurons kept.
    :param reg_factor: Multiplier for L2 Norm of weights.
    :param fc_layers: List of sizes of hidden fully connected layers (excluding the output layer).
    :param var_batch_norm: List of True/False values turning batch normalization on/off for each GC layer.
    :param batch_norm_output: Batch normalization value for the output layer (True/False). Ununsed if var_batch_norm is not None.
    :param var_k: List of GC orders K for each layer. Length = gc_depth.
    :param const_k: Constant K value for each GC layer. Unused if var_k is not None.
    :param stat_layer: Type of statistical layer applied for invariance. Can be None, mean, meanvar, var, or hist.
    :param pooltype: Type of pooling used for GC layers (max or avg).
    :param activation_func: Type of activation function applied for all GC and FC layers (relu, leaky_relu, elu, etc.).
    :param gc_depth: Number of GC layers in the network. Fixed at eight if NSIDE=1024 and pooling by two every layer.
    :param filters: List of # of filters for each GC layer. Length = gc_depth.
    :param batch_size: Batch size for training the network. Ideally a power of two.
    :param nfilters: Constant # of filters for each GC layer. Unused if filters is not None.
    :param exp_name: Experiment ID to define and track directories.
    :param order: HEALPIX order for partial-sky maps. Fixed at 2.
    :param decay_factor: Decay factor by which learning rate gets multiplied every decay_freq steps depending on decay_staircase.
    :param nmaps: Number of full-sky maps from which the training data is being generated.
    :param input_channels: Number of input partial-sky maps. 1 for convergence, 2 for shear, +1 for counts-in-cells.
    :param learning_rate: Initial learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model config

    :return: Params dict for DeepSphere model trained on FLASK v2 data without galaxy-matter bias. Doesn't allow for sophisticated regularization or loss functions.
    """
    params = dict()
    params['dir_name'] = "flaskv2-biasless-{}".format(exp_name)

    # Types of layers.
    params['conv'] = convtype  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = pooltype  # Pooling: max or average.
    params['activation'] = activation_func  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = stat_layer  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.

    if filters is None:
        filters = [nfilters] * gc_depth

    if var_k is None:
        var_k = [const_k] * gc_depth

    if var_batch_norm is None:
        var_batch_norm = [True] * (gc_depth - 1) + [batch_norm_output]

    if nsides is None:
        nsides = [nside // 2 ^ i for i in range(gc_depth + 1)]

    params['F'] = filters  # Graph convolutional layers: number of feature maps.
    params['K'] = var_k  # Polynomial orders.
    params['batch_norm'] = var_batch_norm  # Batch normalization.
    params['M'] = fc_layers + [num_outputs]  # Fully connected layers: output dimensionalities.
    params['input_channel'] = input_channels  # Two channels (spherical maps) per sample.

    # Pooling.
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, order)
    #     params['batch_norm_full'] = []

    # Regularization (to prevent over-fitting).
    params[
        'regularization'] = reg_factor  # Amount of L2 regularization over the weights
    # (will be divided by the number of weights).
    params['dropout'] = dropout_rate  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = num_epochs  # Number of passes through the training data.
    params['batch_size'] = batch_size  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(learning_rate, step, decay_steps=decay_freq,
                                                                  decay_rate=decay_factor, staircase=decay_staircase)
    # params['scheduler'] = lambda step: learning_rate
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['loss'] = loss_func  # Regression loss.

    # Number of model evaluations during training (influence training time).
    params['eval_frequency'] = (12 * order * order * nmaps / batch_size) / 3  # Thrice per epoch

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // order) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // order) ** 2))
        print('=> #pixels for training (input): {:,}'.format(
            params['num_epochs'] * 12 * order * order * nmaps * (NSIDE // order) ** 2))

    return params


def params_by_architecture(architecture, verbose=True, path_to_checkpoints="", num_epochs=20, learning_rate=1e-4,
                           eval_frequency=3, input_channels=None, nmaps=None, decay_factor=0.999, order=ORDER,
                           exp_name=None, nfilters=None, batch_size=16):
    """
    Returns params dict for a specified architecture
    :param batch_size:
    :param nfilters:
    :param exp_name:
    :param order:
    :param decay_factor:
    :param nmaps:
    :param input_channels:
    :param eval_frequency: Evaluation frequency (# of batches)
    :param architecture: Architecture name string
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :return: Params dict for DeepSphere model
    """
    if architecture == "v2":
        return params_v2(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "v8":
        return params_v8(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "v9":
        return params_v9(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "v10":
        return params_v10(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "winning":
        return params_v2(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "two-output":
        return params_v8(verbose=verbose, path_to_checkpoints=path_to_checkpoints)
    if architecture == "v11":
        return params_v11(verbose=verbose, num_epochs=num_epochs,
                          learning_rate=learning_rate, eval_frequency=eval_frequency)
    if architecture == "v12":
        return params_v12(verbose=verbose, num_epochs=num_epochs,
                          learning_rate=learning_rate, eval_frequency=eval_frequency)
    if architecture == "data1":
        return params_vdata1(exp_name, input_channels, nmaps, nfilters, verbose=verbose, num_epochs=num_epochs,
                             learning_rate=learning_rate, decay_factor=decay_factor,
                             order=order, batch_size=batch_size)
        print("Error: Architecture {} not found".format(architecture))

        # Models


def model_by_architecture(architecture, verbose=True, path_to_checkpoints="", num_epochs=20, learning_rate=1e-4,
                          eval_frequency=3, input_channels=None, nmaps=None, decay_factor=0.999, order=ORDER,
                          exp_name=None, nfilters=None, batch_size=16):
    """
    Returns DeepSphere model object for a specified architecture
    :param batch_size:
    :param nfilters:
    :param exp_name:
    :param order:
    :param decay_factor:
    :param nmaps:
    :param input_channels:
    :param eval_frequency: Evaluation frequency (# of batches)
    :param architecture: Architecture name string
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :return: DeepSphere model object
    """
    return models.deepsphere(
        **params_by_architecture(architecture, verbose=verbose, path_to_checkpoints=path_to_checkpoints,
                                 num_epochs=num_epochs, learning_rate=learning_rate, eval_frequency=eval_frequency,
                                 input_channels=input_channels, nmaps=nmaps, decay_factor=decay_factor, order=order,
                                 exp_name=exp_name, nfilters=nfilters, batch_size=batch_size))


def model_v2_biasless(exp_name, convtype='chebyshev5', pooltype='max', nmaps=20,
                       activation_func='relu', stat_layer=None, input_channels=1,
                       gc_depth=8, nfilters=64,
                       const_k=5, var_k=None,
                       filters=None, batch_norm_output=False,
                       var_batch_norm=None,
                       fc_layers=[], num_outputs=1,
                       reg_factor=0, dropout_rate=0.8,
                       verbose=True, num_epochs=20,
                       learning_rate=1e-4,
                       decay_factor=0.999, decay_freq=1,
                       decay_staircase=False, loss_func="l1",
                       nside=NSIDE, nsides=None,
                       order=ORDER, batch_size=16):
    """
    Returns params dict for vdata1 type architectures

    :param convtype: Type of graph convolution performed ("chebyshev5" or "monomials"). TODO: Figure out difference
    :param num_outputs: 1 for just sigma_8, 2 for sigma_8 and predicted log-variance q
    :param nsides: List of NSIDES for graph convolutional layers. Length = gc_depth
    :param nside: NSIDE of input maps. Should be 1024.
    :param loss_func: Choice of loss function ("l1", "custom1", "custom2", "l2"). Must be implemented in DeepSphere codebase.
    :param decay_staircase: If true, performs integer division in lr decay, decaying every decay_freq steps.
    :param decay_freq: If decay_staircase=true, acts to stagger decays. Otherwise, brings down decay factor.
    :param dropout_rate: Percentage of neurons kept. TODO: Figure out what layers this applies to.
    :param reg_factor: Multiplier for L2 Norm of weights. TODO: Figure out good value, implement more complex reg, and allow for adaptive scheduling.
    :param fc_layers: List of sizes of hidden fully connected layers (excluding the output layer).
    :param var_batch_norm: List of True/False values turning batch normalization on/off for each GC layer. TODO: Figure out how this works for FC layers.
    :param batch_norm_output: Batch normalization value for the output layer (True/False). Ununsed if var_batch_norm is not None.
    :param var_k: List of GC orders K for each layer. Length = gc_depth.
    :param const_k: Constant K value for each GC layer. Unused if var_k is not None.
    :param stat_layer: Type of statistical layer applied for invariance. Can be None, mean, meanvar, var, or hist.
    :param pooltype: Type of pooling used for GC layers (max or avg). TODO: Figure out which one is better.
    :param activation_func: Type of activation function applied for all GC and FC layers (relu, leaky_relu, elu, etc.). TODO: Figure out which one is best.
    :param gc_depth: Number of GC layers in the network. Fixed at eight if NSIDE=1024 and pooling by two every layer.
    :param filters: List of # of filters for each GC layer. Length = gc_depth.
    :param batch_size: Batch size for training the network. Ideally a power of two.
    :param nfilters: Constant # of filters for each GC layer. Unused if filters is not None.
    :param exp_name: Experiment ID to define and track directories.
    :param order: HEALPIX order for partial-sky maps. Fixed at 2. TODO: Is 4 better?
    :param decay_factor: Decay factor by which learning rate gets multiplied every decay_freq steps depending on decay_staircase. TODO: What to use?
    :param nmaps: Number of full-sky maps from which the training data is being generated.
    :param input_channels: Number of input partial-sky maps. 1 for convergence, 2 for shear, +1 for counts-in-cells.
    :param learning_rate: Initial learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model config

    :return: DeepSphere model for training on FLASK v2 data without galaxy-matter bias. Doesn't allow for sophisticated regularization or loss functions.
    """

    return models.deepsphere(**params_v2_biasless(exp_name, convtype, pooltype, nmaps,
                                                   activation_func, stat_layer, input_channels,
                                                   gc_depth, nfilters,
                                                   const_k, var_k,
                                                   filters, batch_norm_output,
                                                   var_batch_norm,
                                                   fc_layers, num_outputs,
                                                   reg_factor, dropout_rate,
                                                   verbose, num_epochs,
                                                   learning_rate,
                                                   decay_factor, decay_freq,
                                                   decay_staircase, loss_func,
                                                   nside, nsides,
                                                   order, batch_size))


# Loss Functions

def l1_loss(preds, labels, avg=True):
    """
    Computes L1 loss for some vector of predictions with some vector of ground-truth values
    :param preds: Numpy array of predictions
    :param labels: Numpy array of labels
    :param avg: Computes a single scalar if True
    :return: Either a single scalar loss or a vector of absolute deviations
    """
    assert preds.shape == labels.shape
    if avg:
        return np.mean(np.abs(preds - labels))
    return np.abs(preds - labels)
