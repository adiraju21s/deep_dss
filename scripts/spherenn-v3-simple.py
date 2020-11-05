import os
import tensorflow as tf
import tensorflow.keras as keras
from deep_dss.helpers import *

# Run on GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = "c"
channels = 1
noiseless_m = False
noiseless_kg = False
rand_bias = False
mixed_bias = False

order = 2
nside = 1024

val_set = "TEST"

exp_name = "spherenn-v3-simple-noisy-fixed-gamma"
num_id = 1
checkpoint_path = "../checkpoints/spherenn/{0}/{0}-{1}".format(exp_name, num_id)
checkpoint_dir = "../checkpoints/spherenn/{0}".format(exp_name)
log_dir = "../log/{0}".format(exp_name)
fig_dir = "../figures/{0}".format(exp_name)


def num_cosmologies(dataset):
    if dataset == "TRAINLITE":
        return 16
    if dataset == "TESTLITE":
        return 4
    if dataset == "TEST":
        return 21
    return 45


def generate_reshaped_data(dataset):
    print("Generating data for {0}!".format(dataset))
    num_cosmos = num_cosmologies(dataset)
    data = split_count_and_lensing_maps_by_dataset(dataset, config=config, noiseless_m=noiseless_m,
                                                   noiseless_kg=noiseless_kg, rand_bias=rand_bias,
                                                   mixed_bias=mixed_bias)
    data["x"] = np.reshape(data["x"], (12 * (order ** 2) * num_cosmos, (nside // order) ** 2, channels))
    data["y"] = np.reshape(data["y"], (12 * (order ** 2) * num_cosmos, 1, 1))
    return data


def build_model():
    return keras.Sequential([
        keras.Input(shape=(262144, 1)),
        keras.layers.Conv1D(64, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(128, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(1, 4, strides=4, activation='relu'),
    ])


def train_model_single_dataset(dataset, n_epochs=12, load_model=False, chkpt_path=None, val_data=None):
    train_data = generate_reshaped_data(dataset)
    if val_data is None:
        val_data = generate_reshaped_data(val_set)

    model = build_model()
    if load_model:
        model.load_weights(chkpt_path)
    model.compile(optimizer="adam", loss=tf.keras.losses.MAE, metrics=[])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/{0}-{1}".format(num_id, dataset),
                                                          histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + "-{0}".format(dataset),
                                                     monitor="val_loss", save_weights_only=True, save_best_only=True,
                                                     verbose=1, mode="min")

    model.fit(x=train_data["x"], y=train_data["y"], batch_size=32, epochs=n_epochs,
              validation_data=(val_data["x"], val_data["y"]),
              callbacks=[tensorboard_callback, cp_callback])
    return checkpoint_path + "-{0}".format(dataset)


def train_model():
    val_data = generate_reshaped_data(val_set)
    print("Training on Q1 data:")
    path = train_model_single_dataset("Q1", val_data=val_data)
    print("Training on Q2 data:")
    path = train_model_single_dataset("Q2", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on Q3 data:")
    path = train_model_single_dataset("Q3", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on Q4 data:")
    train_model_single_dataset("Q4", load_model=True, chkpt_path=path, val_data=val_data)


def full_predictions_and_truths():
    model = build_model()
    model.load_weights(checkpoint_path + "-Q4")

    print("Processing Q1 data")
    data = generate_reshaped_data("Q1")
    preds_q1 = model.predict(data["x"])
    truths_q1 = data["y"]
    q1 = {"p": preds_q1, "t": truths_q1}

    print("Processing Q2 data")
    data = generate_reshaped_data("Q2")
    preds_q2 = model.predict(data["x"])
    truths_q2 = data["y"]
    q2 = {"p": preds_q2, "t": truths_q2}

    print("Processing Q3 data")
    data = generate_reshaped_data("Q3")
    preds_q3 = model.predict(data["x"])
    truths_q3 = data["y"]
    q3 = {"p": preds_q3, "t": truths_q3}

    print("Processing Q4 data")
    data = generate_reshaped_data("Q4")
    preds_q4 = model.predict(data["x"])
    truths_q4 = data["y"]
    q4 = {"p": preds_q4, "t": truths_q4}

    print("Processing TEST data")
    data = generate_reshaped_data("TEST")
    preds_test = model.predict(data["x"])
    truths_test = data["y"]
    test = {"p": preds_test, "t": truths_test}

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "TEST": test}


def print_losses(full_results):
    with open("{0}/{1}-{2}-losses.txt".format(fig_dir, exp_name, num_id), "w") as logfile:
        print("Average Q1 Loss:", np.average(np.abs(full_results["Q1"]["p"] - full_results["Q1"]["t"])), file=logfile)
        print("Average Q2 Loss:", np.average(np.abs(full_results["Q2"]["p"] - full_results["Q2"]["t"])), file=logfile)
        print("Average Q3 Loss:", np.average(np.abs(full_results["Q3"]["p"] - full_results["Q3"]["t"])), file=logfile)
        print("Average Q4 Loss:", np.average(np.abs(full_results["Q4"]["p"] - full_results["Q4"]["t"])), file=logfile)
        print("Average TEST Loss:", np.average(np.abs(full_results["TEST"]["p"] - full_results["TEST"]["t"])),
              file=logfile)


train_model()

results = full_predictions_and_truths()

print_losses(results)
