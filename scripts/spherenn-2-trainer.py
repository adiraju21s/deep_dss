import os
import sys
import subprocess
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from deep_dss.helpers import *
import numpy as np
import random

# Run on GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def config_string_to_variables(conf_str):
    split_conf_str = conf_str.split('-')
    s = int(split_conf_str[1])
    conf = split_conf_str[2]
    modifier_num = int(split_conf_str[3])
    run = int(split_conf_str[4])
    fb = modifier_num % 2
    modifier_num = (modifier_num - fb) >> 1
    gl = modifier_num % 2
    modifier_num = (modifier_num - gl) >> 1
    gc = modifier_num % 2
    modifier_num = (modifier_num - gc) >> 1
    nl = modifier_num % 2
    modifier_num = (modifier_num - nl) >> 1
    nc = modifier_num % 2
    modifier_num = (modifier_num - nc) >> 1
    assert modifier_num == 0
    return s, conf, nc, nl, gc, gl, fb, run


def config_variables_to_string(s, conf, nc, nl, gc, gl, fb, run):
    modifier_num = 16 * nc + 8 * nl + 4 * gc + 2 * gl + fb
    return "spherenn-{0}-{1}-{2}-{3}".format(s, conf, modifier_num, run)


def num_cosmologies(dataset):
    if dataset == "TRAINLITE":
        return 16
    if dataset == "TESTLITE":
        return 4
    if dataset == "TEST":
        return 21
    if dataset[0] == "Q":
        return 45
    if dataset == "O1" or dataset == "Q3" or dataset == "Q5" or dataset == "Q7":
        return 22
    return 23


def channels_by_config(conf):
    if conf == "c":
        return 1
    if conf == "k":
        return 1
    if conf == "g":
        return 2
    if conf == "ck":
        return 2
    if conf == "cg":
        return 3
    if conf == "kg":
        return 3
    if conf == "ckg":
        return 4


def outputs_by_config(conf):
    if conf == "c":
        return 2
    if conf == "k":
        return 1
    if conf == "g":
        return 1
    if conf == "ck":
        return 2
    if conf == "cg":
        return 2
    if conf == "kg":
        return 1
    if conf == "ckg":
        return 2


config_string = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
val_set = "TEST"

series, config, noiseless_counts, noiseless_lensing, gaussian_counts, gaussian_lensing, free_bias, run_id = config_string_to_variables(
    config_string)

checkpoint_path = "../spherenn/checkpoints/" + config_string
log_dir = "../spherenn/log/" + config_string
metrics_dir = "../spherenn/metrics/" + config_string

subprocess.call(["mkdir", checkpoint_path])
subprocess.call(["mkdir", log_dir])
subprocess.call(["mkdir", metrics_dir])

channels = channels_by_config(config)
num_outputs = outputs_by_config(config)

np.random.seed(170 * run_id)
random.seed(170 * run_id)
tf.random.set_seed(170 * run_id)

nside = 1024
order = 2
prior_low = 0.94
prior_high = 2.86


def generate_reshaped_data(dataset):
    print("Generating data for {0}!".format(dataset))
    num_cosmos = num_cosmologies(dataset)
    data = split_count_and_lensing_maps_by_dataset(dataset, config=config, noiseless_m=noiseless_counts,
                                                   noiseless_kg=noiseless_lensing,
                                                   free_bias=free_bias, gaussian=(gaussian_counts & gaussian_lensing),
                                                   prior_low=prior_low, prior_high=prior_high)
    data["x"] = np.reshape(data["x"], (12 * (order ** 2) * num_cosmos, (nside // order) ** 2, channels))
    data["y"] = np.reshape(data["y"], (12 * (order ** 2) * num_cosmos, 1, num_outputs))
    return data


def build_model():
    return keras.Sequential([
        keras.Input(shape=(262144, channels)),
        keras.layers.Conv1D(64, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(128, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(256, 4, strides=4, activation='relu'),
        keras.layers.Conv1D(num_outputs, 4, strides=4, activation='relu'),
    ])


def train_model_single_dataset(dataset, load_model=False, chkpt_path=None, val_data=None):
    train_data = generate_reshaped_data(dataset)
    if val_data is None:
        val_data = generate_reshaped_data(val_set)

    model = build_model()
    if load_model:
        model.load_weights(chkpt_path)
    model.compile(optimizer="adam", loss=tf.keras.losses.MAE, metrics=[])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + "-{}".format(dataset),
                                                     monitor="val_loss", save_weights_only=True, save_best_only=True,
                                                     verbose=1, mode="min")

    model.fit(x=train_data["x"], y=train_data["y"], batch_size=batch_size, epochs=epochs,
              validation_data=(val_data["x"], val_data["y"]),
              callbacks=[tensorboard_callback, cp_callback])
    return checkpoint_path + "-{0}".format(dataset)


def train_model():
    val_data = generate_reshaped_data(val_set)
    print("Training on O1 data:")
    path = train_model_single_dataset("O1", val_data=val_data)
    print("Training on O2 data:")
    path = train_model_single_dataset("O2", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O3 data:")
    path = train_model_single_dataset("O3", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O4 data:")
    path = train_model_single_dataset("O4", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O5 data:")
    path = train_model_single_dataset("O5", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O6 data:")
    path = train_model_single_dataset("O6", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O7 data:")
    path = train_model_single_dataset("O7", load_model=True, chkpt_path=path, val_data=val_data)
    print("Training on O8 data:")
    train_model_single_dataset("O8", load_model=True, chkpt_path=path, val_data=val_data)


def full_predictions_and_truths():
    model = build_model()
    model.load_weights(checkpoint_path + "-O8")

    print("Processing Q1 data")
    data = generate_reshaped_data("O1")
    preds_q1 = model.predict(data["x"])
    truths_q1 = data["y"]
    data = generate_reshaped_data("O2")
    preds_q1 = np.concatenate(preds_q1, model.predict(data["x"]))
    truths_q1 = np.concatenate(truths_q1, data["y"])
    q1 = {"p": preds_q1, "t": truths_q1}

    print("Processing Q2 data")
    data = generate_reshaped_data("O3")
    preds_q2 = model.predict(data["x"])
    truths_q2 = data["y"]
    data = generate_reshaped_data("O4")
    preds_q2 = np.concatenate(preds_q2, model.predict(data["x"]))
    truths_q2 = np.concatenate(truths_q2, data["y"])
    q2 = {"p": preds_q2, "t": truths_q2}

    print("Processing Q3 data")
    data = generate_reshaped_data("O5")
    preds_q3 = model.predict(data["x"])
    truths_q3 = data["y"]
    data = generate_reshaped_data("O6")
    preds_q3 = np.concatenate(preds_q3, model.predict(data["x"]))
    truths_q3 = np.concatenate(truths_q3, data["y"])
    q3 = {"p": preds_q3, "t": truths_q3}

    print("Processing Q4 data")
    data = generate_reshaped_data("O7")
    preds_q4 = model.predict(data["x"])
    truths_q4 = data["y"]
    data = generate_reshaped_data("O8")
    preds_q4 = np.concatenate(preds_q4, model.predict(data["x"]))
    truths_q4 = np.concatenate(truths_q4, data["y"])
    q4 = {"p": preds_q4, "t": truths_q4}

    print("Processing TEST data")
    data = generate_reshaped_data("TEST")
    preds_test = model.predict(data["x"])
    truths_test = data["y"]
    test = {"p": preds_test, "t": truths_test}

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "TEST": test}


def serialize_predictions(full_results):
    pickle.dump(full_results, open("{0}/{1}-full-preds.pkl".format(metrics_dir, config_string), "wb"))


def load_predictions():
    return pickle.load(open("{0}/{1}-full-preds.pkl".format(metrics_dir, config_string), "rb"))


def print_losses(full_results):
    with open("{0}/{1}-losses.txt".format(metrics_dir, config_string), "w") as logfile:
        print("Average Q1 Loss:", np.average(np.abs(full_results["Q1"]["p"] - full_results["Q1"]["t"])), file=logfile)
        print("Average Q2 Loss:", np.average(np.abs(full_results["Q2"]["p"] - full_results["Q2"]["t"])), file=logfile)
        print("Average Q3 Loss:", np.average(np.abs(full_results["Q3"]["p"] - full_results["Q3"]["t"])), file=logfile)
        print("Average Q4 Loss:", np.average(np.abs(full_results["Q4"]["p"] - full_results["Q4"]["t"])), file=logfile)
        print("Average TEST Loss:", np.average(np.abs(full_results["TEST"]["p"] - full_results["TEST"]["t"])),
              file=logfile)


def bias_and_variance_by_cosmo(full_results):
    all_truths = np.concatenate((full_results["Q1"]["t"], full_results["Q2"]["t"], full_results["Q3"]["t"],
                                 full_results["Q4"]["t"], full_results["TEST"]["t"]))
    all_preds = np.concatenate((full_results["Q1"]["p"], full_results["Q2"]["p"], full_results["Q3"]["p"],
                                full_results["Q4"]["p"], full_results["TEST"]["p"]))
    biases = np.zeros((201, num_outputs))
    variances = np.zeros((201, num_outputs))
    if num_outputs == 1:
        all_truths_s8 = np.reshape(all_truths, (201 * 12 * order ** 2))
        all_preds_s8 = np.reshape(all_preds, (201 * 12 * order ** 2))
        for i in range(201):
            s8 = round(0.5 + i * (1.2 - 0.5) / 200, 5)
            sel_pred = all_preds_s8[np.where(all_truths_s8 == s8)]
            biases[i, 0] = np.mean(sel_pred) - s8
            variances[i, 0] = np.var(sel_pred)
    if num_outputs == 2:
        all_truths_s8 = np.reshape(all_truths[:, :, 0], (201 * 12 * order ** 2))
        all_preds_s8 = np.reshape(all_preds[:, :, 0], (201 * 12 * order ** 2))
        all_truths_b = np.reshape(all_truths[:, :, 1], (201 * 12 * order ** 2))
        all_preds_b = np.reshape(all_preds[:, :, 1], (201 * 12 * order ** 2))
        for i in range(201):
            s8 = round(0.5 + i * (1.2 - 0.5) / 200, 5)
            sel_pred_s8 = all_preds_s8[np.where(all_truths_s8 == s8)]
            biases[i, 0] = np.mean(sel_pred_s8) - s8
            variances[i, 0] = np.var(sel_pred_s8)
        for i in range(48):
            b = round(prior_low + i * (prior_high - prior_low) / (12 * order ** 2), 2)
            sel_pred_b = all_preds_b[np.where(all_truths_b == b)]
            biases[i, 1] = np.mean(sel_pred_b) - b
            variances[i, 1] = np.var(sel_pred_b)
    pickle.dump({"b": biases, "v": variances},
                open("{0}/{1}-bias-variance.pkl".format(metrics_dir, config_string), "wb"))


train_model()

results = full_predictions_and_truths()

print_losses(results)

serialize_predictions(results)

bias_and_variance_by_cosmo(results)
