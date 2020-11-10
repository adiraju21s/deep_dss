import os
import sys

from deep_dss.helpers import *
from deep_dss.models import *

import numpy as np
from deepsphere.data import LabeledDataset

# Run on GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def total_channels(c):
    if c[0] == "c":
        return 1 + lensing_channels(c[1:])
    return lensing_channels(c)


name = sys.argv[1]
config = sys.argv[2]
channels = total_channels(config)
order = 2
start_level = int(sys.argv[3])
start_it = int(sys.argv[4])
ilr = float(sys.argv[5])
gaussian = sys.argv[6]
decay_noise = 0.8
decay_train = 0.999
duration = 1
noise_levels = 4
threshold = 0.06
epoch_size = 48  # Number of batches per epoch


# nlevels includes the noiseless level!
def density_count_by_iter(noise_level, nlevels=6):
    noise_scales = np.logspace(np.log10(1000), np.log10(0.04377), num=nlevels)
    return noise_scales[noise_level]


def noiseless_count_by_iter(noise_level):
    return noise_level == 0


# nlevels includes the noiseless level!
def density_kg_by_iter(noise_level, initial_width=0, target_width=0.025, nlevels=6, sigma_e=0.25, nside=1024):
    if noise_level == 0:
        return 1000
    sigma = initial_width + noise_level * (target_width - initial_width) / (nlevels - 1)
    return sigma_e ** 2 / (2 * hp.nside2pixarea(nside, degrees=True) * 3600 * (sigma ** 2))


def noiseless_kg_by_iter(noise_level):
    return noise_level == 0


def train_one_epoch(lr, noise_level, iteration):
    train_dict = split_count_and_lensing_maps_by_dataset("TRAINLITE", config=config, order=order,
                                                         density_m=density_count_by_iter(noise_level,
                                                                                         nlevels=noise_levels),
                                                         noiseless_m=noiseless_count_by_iter(noise_level),
                                                         density_kg=density_kg_by_iter(noise_level,
                                                                                       nlevels=noise_levels),
                                                         noiseless_kg=noiseless_kg_by_iter(noise_level),
                                                         scramble=True, gaussian=(gaussian == "GAUSS"))

    train = LabeledDataset(train_dict["x"], train_dict["y"])

    val_dict = split_count_and_lensing_maps_by_dataset("TESTLITE", config=config, order=order,
                                                       density_m=density_count_by_iter(noise_level,
                                                                                       nlevels=noise_levels),
                                                       noiseless_m=noiseless_count_by_iter(noise_level),
                                                       density_kg=density_kg_by_iter(noise_level,
                                                                                     nlevels=noise_levels),
                                                       noiseless_kg=noiseless_kg_by_iter(noise_level),
                                                       scramble=True, gaussian=(gaussian == "GAUSS"))

    val = LabeledDataset(val_dict["x"], val_dict["y"])

    model = model_v3(exp_name="{0}-{1}-{2}".format(name, config, gaussian),
                     gc_depth=12, activation_func="leaky_relu", input_channels=channels,
                     nsides=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 32, 16, 8, 4],
                     filters=[32] * 6 + [64] * 6, var_k=[5] * 6 + [10] * 6,
                     fc_layers=[128], learning_rate=lr)

    if noise_level == 0 and iteration == 0:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                session=model._get_session())

    np.savez_compressed(
        "../metrics/v3-{0}-{1}-{2}-{3}-{4}.npz".format(name, config, gaussian, noise_level, iteration),
        lval=loss_validation,
        ltrain=loss_training, t=t_step)

    print("-----------------END OF EPOCH-----------------")
    print("NOISE LEVEL: {0}, ITERATION: {1}, LR: {2}".format(noise_level, iteration, lr))
    print("VALIDATION LOSS (used for curriculum strategy): ", loss_validation)
    print("----------------------------------")

    return np.mean(loss_validation)


def learning_rate(n, iteration, initial_rate=ilr, nsteps=epoch_size, decay_factor_noise=decay_noise,
                  decay_factor_train=decay_train):
    return initial_rate * (decay_factor_noise ** n) * (decay_factor_train ** (iteration * nsteps))


curr_dur = 0
for i in range(start_level, noise_levels):
    if i == start_level:
        it = start_it
    else:
        it = 0
    while curr_dur < duration:
        curr_dur = 0
        loss = train_one_epoch(learning_rate(i, it), i, it)
        if loss < threshold:
            curr_dur = curr_dur + 1
        it = it + 1
