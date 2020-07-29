from deep_dss.utils import set_constants, set_paths, MACHINE
from deepsphere import utils, models
import tensorflow as tf
import numpy as np

(NSIDE, NPIX, PIXEL_AREA, ORDER, BIAS, DENSITY_M, DENSITY_KG, ELLIP_SIGMA) = set_constants()

(PATH_TO_INPUT, PATH_TO_OUTPUT, PATH_TO_CHECKPOINTS, PATH_TO_VAL) = set_paths(MACHINE)


# Model parameters

def params_v2(verbose=True, path_to_checkpoints=""):
    """
    Returns params dict for v2 ("winning") architecture
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = path_to_checkpoints + "flask-101-v2"

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
    params['dir_name'] = path_to_checkpoints + "flask-101-v8"

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
    params['dir_name'] = path_to_checkpoints + "flask-101-v9"

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
    params['dir_name'] = path_to_checkpoints + "flask-101-v10"

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


def params_v11(verbose=True, num_epochs=20, learning_rate=1e-4):
    """
    Returns params dict for v11 architecture
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask-101-v11"

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
    params['eval_frequency'] = 60

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_v12(verbose=True, num_epochs=20, learning_rate=1e-4):
    """
    Returns params dict for v12 architecture
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :param verbose: Outputs information on model configuration
    :return: Params dict for DeepSphere model
    """
    params = dict()
    params['dir_name'] = "flask-101-v12"

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
    params['eval_frequency'] = 60

    if verbose:
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([(nside // ORDER) ** 2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size'] * (NSIDE // ORDER) ** 2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs'] * 422 * (NSIDE // ORDER) ** 2))

    return params


def params_by_architecture(architecture, verbose=True, path_to_checkpoints="", num_epochs=20, learning_rate=1e-4):
    """
    Returns params dict for a specified architecture
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
                          learning_rate=learning_rate)
    if architecture == "v12":
        return params_v12(verbose=verbose, num_epochs=num_epochs,
                          learning_rate=learning_rate)
    print("Error: Architecture {} not found".format(architecture))


# Models

def model_by_architecture(architecture, verbose=True, path_to_checkpoints="", num_epochs=20, learning_rate=1e-4):
    """
    Returns DeepSphere model object for a specified architecture
    :param architecture: Architecture name string
    :param verbose: Outputs information on model configuration
    :param path_to_checkpoints: Path to parent of checkpoints directory (include '/'!)
    :param learning_rate: Constant learning rate to use during training
    :param num_epochs: Number of epochs for training the model
    :return: DeepSphere model object
    """
    return models.deepsphere(
        **params_by_architecture(architecture, verbose=verbose, path_to_checkpoints=path_to_checkpoints,
                                 num_epochs=num_epochs, learning_rate=learning_rate))


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
