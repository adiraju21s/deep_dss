import os

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


config = "g"
channels = total_channels(config)
order = 2
nmaps = 10
k = 64
epochs1 = 20
epochs2 = 10
lr1 = 1e-4
lr2 = 2e-5

val_dict = split_count_and_lensing_maps_by_dataset("TEST", config=config, order=order,
                                                   scramble=True)
val_dict["x"] = val_dict["x"][:64]
val_dict["y"] = val_dict["y"][:64]

val = LabeledDataset(val_dict["x"], val_dict["y"])

train_dict = split_count_and_lensing_maps_by_dataset("Q1", config=config, noiseless_m=True,
                                                     noiseless_kg=True, order=order,
                                                     scramble=True)
train_dict["x"] = train_dict["x"][:nmaps * 12 * order * order]
train_dict["y"] = train_dict["y"][:nmaps * 12 * order * order]

train = LabeledDataset(train_dict["x"], train_dict["y"])

model = model_by_architecture("data1", num_epochs=epochs1, learning_rate=lr1, input_channels=channels, nmaps=nmaps,
                              order=order, exp_name="shear", nfilters=k)

accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)

np.savez_compressed("vdata1-shear-metrics-1.npz", lval=loss_validation, ltrain=loss_training, t=t_step)

train_dict = split_count_and_lensing_maps_by_dataset("Q1", config=config, order=order,
                                                     scramble=True)
train_dict["x"] = train_dict["x"][:nmaps * 12 * order * order]
train_dict["y"] = train_dict["y"][:nmaps * 12 * order * order]

train = LabeledDataset(train_dict["x"], train_dict["y"])

model = model_by_architecture("data1", num_epochs=epochs2, learning_rate=lr2, input_channels=channels, nmaps=nmaps,
                              order=order, exp_name="shear", nfilters=k)

accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val, session=model._get_session())

np.savez_compressed("vdata1-shear-metrics-2.npz", lval=loss_validation, ltrain=loss_training, t=t_step)
