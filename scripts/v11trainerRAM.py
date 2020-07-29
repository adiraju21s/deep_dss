import os

from deep_dss.utils import *
from deep_dss.models import *

import numpy as np
from deepsphere.data import LabeledDataset

# Run on GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH_TO_OUTPUT = "/pylon5/ch4s8kp/adiraj21/flaskwrapper/output/"
PATH_TO_VAL = "/pylon5/ch4s8kp/adiraj21/DeepSphere/validation_101.npz"

y = {}
for dataset_name in dataset_names():
    y[dataset_name] = cosmologies_list(dataset_name)

val = LabeledDataset(np.load(PATH_TO_VAL)['arr_0'][:, :, 0],
                     np.load(PATH_TO_VAL)['arr_1'])

model = model_by_architecture("v11", num_epochs=1, learning_rate=1e-4, eval_frequency=6)

for epoch in range(16):
    for i in range(4):
        for j in range(5):
            if (5 * i + j) % 4 == 0: model_by_architecture("v11", num_epochs=1, learning_rate=1e-4)
            y_train = y["Q{}".format(i + 1)][4 * j:4 * (j + 1)]
            train = split_poisson_maps_by_vals(y_train, noiseless=True, multiplier=1.0, deepsphere_dataset=True)
            if i + j == 0:
                accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val)
            else:
                accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                        session=model._get_session())

model = model_by_architecture("v11", num_epochs=1, learning_rate=8e-5, eval_frequency=6)

for epoch in range(8):
    for i in range(4):
        for j in range(5):
            if (5 * i + j) % 4 == 0: model_by_architecture("v11", num_epochs=1, learning_rate=1e-4)
            y_train = y["Q{}".format(i + 1)][4 * j:4 * (j + 1)]
            train = split_poisson_maps_by_vals(y_train, density=0.4, multiplier=1.0, deepsphere_dataset=True)
            accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                    session=model._get_session())

model = model_by_architecture("v11", num_epochs=1, learning_rate=6e-5, eval_frequency=6)

for epoch in range(8):
    for i in range(4):
        for j in range(5):
            if (5 * i + j) % 4 == 0: model_by_architecture("v11", num_epochs=1, learning_rate=1e-4)
            y_train = y["Q{}".format(i + 1)][4 * j:4 * (j + 1)]
            train = split_poisson_maps_by_vals(y_train, density=0.2, multiplier=1.0, deepsphere_dataset=True)
            accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                    session=model._get_session())

model = model_by_architecture("v11", num_epochs=1, learning_rate=2e-5)

for epoch in range(8):
    for i in range(4):
        for j in range(5):
            if (5 * i + j) % 4 == 0: model_by_architecture("v11", num_epochs=1, learning_rate=1e-4)
            y_train = y["Q{}".format(i + 1)][4 * j:4 * (j + 1)]
            train = split_poisson_maps_by_vals(y_train, density=0.04377, multiplier=1.0, deepsphere_dataset=True)
            accuracy_validation, loss_validation, loss_training, t_step = model.fit(train, val,
                                                                                    session=model._get_session())
