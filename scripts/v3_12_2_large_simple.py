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
lr = float(sys.argv[5])

train_dict = split_count_and_lensing_maps_by_dataset("TRAINLITE", config=config, order=order,
                                                     noiseless_m=True, noiseless_kg=True,
                                                     scramble=True)

train = LabeledDataset(train_dict["x"], train_dict["y"])

val_dict = split_count_and_lensing_maps_by_dataset("TESTLITE", config=config, order=order,
                                                   noiseless_m=True, noiseless_kg=True,
                                                   scramble=True)

val = LabeledDataset(val_dict["x"], val_dict["y"])

model = model_v3(exp_name="simple-{0}-{1}".format(name, config),
                 gc_depth=12, input_channels=channels, num_epochs=12,
                 nsides=[1024, 1024, 512, 512, 256, 256, 128, 128, 64, 32, 16, 8, 4],
                 nfilters=64, const_k=10,
                 fc_layers=[128], learning_rate=lr)

accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)

np.savez_compressed(
    "../metrics/v3-simple-{0}-{1}-noiseless.npz".format(name, config),
    lval=loss_validation,
    ltrain=loss_training, t=t_step)
