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
lr = float(sys.argv[3])


def train_on_dataset(dataset, reload=False):
    train_dict = split_count_and_lensing_maps_by_dataset(dataset, config=config, order=order,
                                                         noiseless_m=True, noiseless_kg=True,
                                                         scramble=True)

    train = LabeledDataset(train_dict["x"], train_dict["y"])

    val_dict = split_count_and_lensing_maps_by_dataset("TESTLITE", config=config, order=order,
                                                       noiseless_m=True, noiseless_kg=True,
                                                       scramble=True)

    val = LabeledDataset(val_dict["x"], val_dict["y"])

    model = model_v3(exp_name="simple-{0}-{1}".format(name, config), gc_depth=16, input_channels=channels,
                     nsides=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4],
                     filters=[32] * 8 + [64] * 8, var_k=[5] * 8 + [10] * 8, num_epochs=12,
                     fc_layers=[128, 128], learning_rate=lr)
    if reload:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                session=model._get_session())
    else:
        accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)

    np.savez_compressed(
        "../metrics/simple-{0}-{1}-{2}-noiseless.npz".format(name, config, dataset),
        lval=loss_validation,
        ltrain=loss_training, t=t_step)


train_on_dataset("Q1")
train_on_dataset("Q2", reload=True)
train_on_dataset("Q3", reload=True)
train_on_dataset("Q4", reload=True)
