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

step = int(sys.argv[1])

val_dict = split_count_and_lensing_maps_by_dataset("TEST", config=config, order=order,
                                                   scramble=True)
val_dict["x"] = val_dict["x"][:64]
val_dict["y"] = val_dict["y"][:64]

val = LabeledDataset(val_dict["x"], val_dict["y"])


def train_one_quartile_epoch(quartile, lr, iteration):
    train_dict = split_count_and_lensing_maps_by_dataset(quartile, config=config, order=order,
                                                         scramble=True)

    train = LabeledDataset(train_dict["x"], train_dict["y"])

    model = model_by_architecture("data1", num_epochs=1, learning_rate=lr, input_channels=channels, nmaps=45,
                                  order=order, exp_name="final2-noisy", nfilters=k)

    if iteration == 1 and quartile == "Q1":
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                session=model._get_session())

    np.savez_compressed("../metrics/vdata1-final2-noisy-metrics-{0}-{1}.npz".format(iteration, quartile), lval=loss_validation,
                        ltrain=loss_training, t=t_step)


def learning_rate(n, q, initial_rate=1e-4, epoch_decay_factor=0.999 ** 539, quartile_decay_factor=0.999 ** 134):
    return initial_rate * epoch_decay_factor ** n * quartile_decay_factor ** q


train_one_quartile_epoch("Q1", learning_rate(step, 1), step)
train_one_quartile_epoch("Q2", learning_rate(step, 2), step)
train_one_quartile_epoch("Q3", learning_rate(step, 3), step)
train_one_quartile_epoch("Q4", learning_rate(step, 4), step)
