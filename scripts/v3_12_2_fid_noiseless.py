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


config = sys.argv[1]
channels = total_channels(config)
order = 2
lr = float(sys.argv[2])
num_epochs = int(sys.argv[3])
gaussian = sys.argv[4]


def train_on_dataset(dataset, reload=False):
    train_dict = split_count_and_lensing_maps_by_dataset(dataset, config=config, order=order,
                                                         noiseless_m=True, noiseless_kg=True,
                                                         scramble=True, gaussian=(gaussian == "GAUSS"))

    train = LabeledDataset(train_dict["x"], train_dict["y"])

    val_dict = split_count_and_lensing_maps_by_dataset(dataset, config=config, order=order,
                                                       noiseless_m=True, noiseless_kg=True,
                                                       scramble=True, gaussian=(gaussian == "GAUSS"))

    val = LabeledDataset(val_dict["x"], val_dict["y"])

    model = model_v3(exp_name="simple-noiseless-2-{0}-{1}".format(config, gaussian),
                     gc_depth=12, input_channels=channels, num_epochs=num_epochs,
                     nsides=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 32, 16, 8, 4],
                     filters=[32] * 6 + [64] * 6, var_k=[5] * 6 + [10] * 6,
                     fc_layers=[128], learning_rate=lr)

    if reload:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                sesssion=model._get_session())
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)

    np.savez_compressed(
        "../metrics/v3-simple-noiseless-2-{0}-{1}.npz".format(config, gaussian),
        lval=loss_validation,
        ltrain=loss_training, t=t_step)


train_on_dataset("Q1")
train_on_dataset("Q2", reload=True)
train_on_dataset("Q3", reload=True)
train_on_dataset("Q4", reload=True)
