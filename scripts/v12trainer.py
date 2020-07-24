import os

from deep_dss.utils import *
from deep_dss.models import *

import numpy as np
from deepsphere.data import LabeledDataset

# Run on GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = split_poisson_maps_by_dataset("Q1", path_to_output="$SCRATCH/flaskwrapper/output", scramble=True, noiseless=True,
                                     deepsphere_dataset=True)
val = LabeledDataset(np.load("$SCRATCH/DeepSphere/validation_101.npz")['arr_0'][:, :, 0],
                     np.load("$SCRATCH/DeepSphere/validation_101.npz")['arr_1'])
model = model_by_architecture("v12", path_to_checkpoints="", num_epochs=16, learning_rate=1e-4)
accuracy_validation, loss_validation, loss_training, t_step = model.fit(data, val)

data = split_poisson_maps_by_dataset("Q1", path_to_output="$SCRATCH/flaskwrapper/output", scramble=True, density=0.4,
                                     deepsphere_dataset=True)
model = model_by_architecture("v12", path_to_checkpoints="", num_epochs=8, learning_rate=8e-5)
accuracy_validation, loss_validation, loss_training, t_step = model.fit(data, val, session=model._get_session())

data = split_poisson_maps_by_dataset("Q1", path_to_output="$SCRATCH/flaskwrapper/output", scramble=True, density=0.2,
                                     deepsphere_dataset=True)
model = model_by_architecture("v12", path_to_checkpoints="", num_epochs=8, learning_rate=6e-5)
accuracy_validation, loss_validation, loss_training, t_step = model.fit(data, val, session=model._get_session())

data = split_poisson_maps_by_dataset("Q1", path_to_output="$SCRATCH/flaskwrapper/output", scramble=True,
                                     deepsphere_dataset=True)
model = model_by_architecture("v12", path_to_checkpoints="", num_epochs=8, learning_rate=2e-5)
accuracy_validation, loss_validation, loss_training, t_step = model.fit(data, val, session=model._get_session())
