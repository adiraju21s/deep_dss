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


config = "k"
channels = total_channels(config)
order = 2

start_level = int(sys.argv[1])
start_it = int(sys.argv[2])
noise_levels = int(sys.argv[3])
ilr = float(sys.argv[4])
decay_noise = float(sys.argv[5])
decay_train = float(sys.argv[6])
duration = 1
threshold = 0.04


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
                                                         density_kg=density_kg_by_iter(noise_level),
                                                         noiseless_kg=noiseless_kg_by_iter(noise_level),
                                                         scramble=True)

    train = LabeledDataset(train_dict["x"], train_dict["y"])

    val_dict = split_count_and_lensing_maps_by_dataset("TESTLITE", config=config, order=order,
                                                       density_kg=density_kg_by_iter(noise_level),
                                                       noiseless_kg=noiseless_kg_by_iter(noise_level),
                                                       scramble=True)

    val = LabeledDataset(val_dict["x"], val_dict["y"])

    model = model_v2_biasless(exp_name="1-fid-leaky-relu", gc_depth=12,
                              nsides=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 32, 16, 8, 4],
                              filters=[32] * 6 + [64] * 6, var_k=[5] * 6 + [10] * 6,
                              fc_layers=[128], learning_rate=lr, activation_func="leaky_relu")

    if noise_level == 0 and iteration == 0:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                session=model._get_session())

    np.savez_compressed("../metrics/v2-biasless-1-fid-leaky-relu-{0}-{1}.npz".format(noise_level, iteration),
                        lval=loss_validation,
                        ltrain=loss_training, t=t_step)

    print("-----------------END OF EPOCH-----------------")
    print("NOISE LEVEL: {0}, ITERATION: {1}, LR: {2}".format(noise_level, iteration, lr))
    print("VALIDATION LOSS (used for curriculum strategy): ", loss_validation)
    print("----------------------------------")

    return np.mean(loss_validation)


def learning_rate(n, initial_rate=ilr, decay_factor=decay_noise):
    return initial_rate * (decay_factor ** n)


curr_dur = 0
for i in range(start_level, noise_levels + 1):
    while curr_dur < duration:
        if i == start_level:
            it = start_it
        else:
            it = 0
        curr_dur = 0
        loss = train_one_epoch(learning_rate(i), i, it)
        if loss < threshold:
            curr_dur = curr_dur + 1
        it = it + 1
