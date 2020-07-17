from deep_dss.utils import set_constants
from deepsphere import utils, models
import tensorflow as tf

(NSIDE, NPIX, PIXEL_AREA, ORDER, BIAS, DENSITY_M, DENSITY_KG, ELLIP_SIGMA) = set_constants()


# Model parameters

def params_v8(verbose=True):
    """
    Returns params dict for v8 architecture
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask-101-v8"

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


def params_v9(verbose=True):
    """
    Returns params dict for v9 architecture
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask-101-v9"

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


def params_v10(verbose=True):
    """
    Returns params dict for v10 architecture
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask-101-v10"

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


def params_by_architecture(architecture, verbose=True):
    """
    Returns params dict for a specified architecture
    :param architecture: Architecture name string
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    if architecture == "v8":
        return params_v8(verbose=verbose)
    if architecture == "v9":
        return params_v9(verbose=verbose)
    if architecture == "v10":
        return params_v10(verbose=verbose)
    print("Error: Architecture {} not found".format(architecture))


# Models

def model_by_architecture(architecture, verbose=True):
    """
    Returns DeepSphere model object for a specified architecture
    :param architecture: Architecture name string
    :param verbose: Outputs information on model configuration
    :return: DeepSphere model object
    """
    return models.deepsphere(**params_by_architecture(architecture, verbose=verbose))
