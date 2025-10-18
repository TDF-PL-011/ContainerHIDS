import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.layers import Input, Dense
from keras.models import Model
from keras import losses, optimizers, regularizers
from chids.shared.misc import _format_input
from chids.conf.config import *
from chids.shared.constants import *
import random
import tensorflow as tf
import keras
from tensorflow.keras import layers as L, regularizers as R


class Training:

    def __init__(self, anomaly_vectors):
        self.anomaly_vectors = anomaly_vectors

    def _autoencoder_model(self, vectors):
        T, F = vectors.shape[1], vectors.shape[2]
        inputs = L.Input(shape=(T, F))
        x = L.Conv1D(64, 3, padding="causal")(inputs)
        x = L.ReLU()(x)
        x = L.Conv1D(64, 3, dilation_rate=2, padding="causal")(x)
        x = L.ReLU()(x)
        x = L.Conv1D(BOTTLENECK, 3, dilation_rate=4, padding="causal")(x)
        z = L.GlobalAveragePooling1D()(x)

        x = L.RepeatVector(T)(z)
        x = L.Conv1DTranspose(64, 3, padding="same")(x)
        x = L.ReLU()(x)
        x = L.Conv1DTranspose(64, 3, padding="same")(x)
        outputs = L.TimeDistributed(L.Dense(F, activation="linear"))(x)

        model = tf.keras.Model(inputs, outputs)
        return model

    def _get_thresholds_list(self, model, inp):
        thresh_list = []
        prediction = model.predict(inp, verbose=VERBOSE)
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[2])
        inp = inp.reshape(inp.shape[0], inp.shape[2])
        training_reconstruction_errors = np.mean(np.square(prediction - inp), axis=1)

        for _theta in np.arange(0.2, 2.2, 0.2):
            thresh_list.append(max(training_reconstruction_errors) * _theta)
        return thresh_list

    def train_model(self):
        # Set seeds for reproducibility (choose a fixed integer, e.g., 42)
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        formatted_anomaly_vectors = _format_input(self.anomaly_vectors)
        training_anomaly_vectors = formatted_anomaly_vectors.reshape(
            formatted_anomaly_vectors.shape[0], 1, formatted_anomaly_vectors.shape[1]
        )
        model = self._autoencoder_model(training_anomaly_vectors)
        optimizer = optimizers.get(OPTIMIZER)
        loss = losses.get(LOSS_FUNC)
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(
            training_anomaly_vectors,
            training_anomaly_vectors,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=VERBOSE,
        )
        thresh_list = self._get_thresholds_list(model, training_anomaly_vectors)
        return thresh_list, model
