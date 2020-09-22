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
k = 64

min_or_max = sys.argv[1]
threshold = float(sys.argv[2])
duration = int(sys.argv[3])


def density_kg_by_iter(noise_level, initial_width=0, delta_width=0.005, sigma_k=0.25, nside=1024):
    sigma = initial_width + delta_width * noise_level
    return sigma_k ** 2 / (2 * hp.nside2pixarea(nside) * 3600 * (sigma ** 2))


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

    model = model_by_architecture("data1", num_epochs=1, learning_rate=lr, decay_factor=1, input_channels=channels,
                                  nmaps=8,
                                  order=order, exp_name="adaptive1-1", nfilters=k)

    if noise_level == 0 and iteration == 0:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                session=model._get_session())

    np.savez_compressed("../metrics/adaptive1-1-metrics-{0}-{1}.npz".format(noise_level, iteration),
                        lval=loss_validation,
                        ltrain=loss_training, t=t_step)

    if min_or_max == "min":
        return np.min(loss_validation)
    else:
        return np.max(loss_validation)


def learning_rate(n, initial_rate=1e-4, decay_factor=0.8):
    return initial_rate * (decay_factor ** n)


curr_dur = 0
for i in range(6):
    while curr_dur < duration:
        it = 0
        curr_dur = 0
        loss = train_one_epoch(learning_rate(i), i, it)
        if loss < threshold:
            curr_dur = curr_dur + 1
        it = it + 1
