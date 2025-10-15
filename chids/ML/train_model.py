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


class Training:

    def __init__(self, anomaly_vectors):
        self.anomaly_vectors = anomaly_vectors

    def _autoencoder_model(self, vectors):
        inputs = Input(shape=(vectors.shape[1], vectors.shape[2]))

        # Enhanced encoder with batch normalization and LeakyReLU
        L1 = Dense(ENCODING_DIM * 2, activation="linear")(inputs)
        L1 = keras.layers.LeakyReLU(alpha=0.1)(L1)
        L1 = keras.layers.BatchNormalization()(L1)
        L1 = keras.layers.Dropout(0.2)(L1)

        L2 = Dense(ENCODING_DIM, activity_regularizer=regularizers.l1(REG_RATE))(L1)
        L2 = keras.layers.LeakyReLU(alpha=0.1)(L2)
        L2 = keras.layers.BatchNormalization()(L2)

        L3 = Dense(BOTTLENECK)(L2)
        L3 = keras.layers.LeakyReLU(alpha=0.1)(L3)

        # Enhanced decoder
        L4 = Dense(ENCODING_DIM)(L3)
        L4 = keras.layers.LeakyReLU(alpha=0.1)(L4)
        L4 = keras.layers.BatchNormalization()(L4)

        L5 = Dense(ENCODING_DIM * 2)(L4)
        L5 = keras.layers.LeakyReLU(alpha=0.1)(L5)

        output = Dense(vectors.shape[2], activation=ACTIVATION)(L5)
        model = Model(inputs=inputs, outputs=output)
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
