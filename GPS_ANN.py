__author__ = "Alexander Timmermann"
__email__ = "alexandertimmermann93@gmail.com"
__status__ = "Production"

######################################################
##### Skript für die Hyperparameteroptimierung
##### des Feedfoward KNNs
######################################################

import numpy as np
import pandas as pd
from keras import optimizers
import sklearn.model_selection
import keras as k
import sys
sys.path.append("../")
from Data_preprocessing import data_preprocess
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import tensorflow as tf
import time as t
import shutil
import pickle
import os
from callbacks import StopTraining
from keras import callbacks
import time
from RNN_Train import save_pickle, load_pickle


# 'Initialize all Parameters and folder paths'

lrate = Real(low=1e-12, high=5e-2, prior='log-uniform', name='lrate')
batch_size = Integer(low=32, high=512, name='batch_size')
beta1 = Real(low=85e-2, high=95e-2, prior='log-uniform', name='beta1')
beta2 = Real(low=90e-2, high=0.99999, prior='log-uniform', name='beta2')
hidden_units=Integer(low=300, high=1500, name='hidden_units')
layer=Integer(low=1, high=5, name='layer')
dimensions = [lrate, beta1, beta2, batch_size, hidden_units, layer]
display_step = 10
dropout_prob = 1.0
momentum = 2.0
dropout = 1.0
training_iters = 1
counter = 0
default_parameters = [1e-3, 0.9, 0.999, 32, 300, 1]
no_bayes_iter = 100
epochs = 100

"""
path = "C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\Studienarbeit_Alexander_Timmermann\\" \
           "Phyton\\Daten\\Messdaten\\Fahrten_mit_SV_info\\null_as_replace_for_missing\\" \
           "kompletter_Datensatz_PPP_N_null_as_replace_a.csv"

folder_name = "MLP_null_replaced_missing"
"""

path = "C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\Studienarbeit_Alexander_Timmermann\\" \
       "Phyton\\Daten\\Messdaten\\Fahrten_mit_SV_info\\kompletter_Datensatz_PPP_N_a.csv"


folder_name = "MLP_replaced_by_ppp_data"
""""
path = "C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\Studienarbeit_Alexander_Timmermann\\" \
       "Phyton\\Daten\\Messdaten\\Fahrten_mit_SV_info\\kompletter_Datensatz_PPP_a.csv"


folder_name = "MLP_only_ppp"
"""

data = data_preprocess(path)
data.normalization_of_data()


@use_named_args(dimensions=dimensions)
def trainiere_knn(lrate, beta1, beta2,  batch_size, hidden_units,
                  layer):
    """

    Training des neuronalen Netzes, die Parameter die ausprobiert werden im Rahmen der Bayes suche, sind die Parameter
    die übergeben werden

    :param lrate: learning Rate
    :param batch_size: groesse der Batch
    :param hidden_units: Anzahl der Hidden Units
    :param beta1: Gewichtsverfall 1. Ordnung
    :param beta2: Gewichtsverfall 2. Ordnung
    :return: die Genauigkeit auf dem Evaluierungsdatensatz wird zurückgeben für die Bayes suche
    """
    global counter
    model_folder = "..\\Daten\\Modelle\\" + folder_name + "\\"
    log_folder = "..\\Daten\\Logs\\" + folder_name + "\\"

    print("Create callbacks")
    stopCallback = StopTraining()
    str_learning_rate = "lrate_{:.3f}e-5".format(lrate * 100000)
    data_name = str_learning_rate + "Units_" + str(hidden_units) + "Layers_" + str(layer)
    checkpoint = callbacks.ModelCheckpoint(model_folder + str(counter) +
                                           "loss_{val_loss:.2f}.hdf5", monitor='val_loss',
                                           verbose=0, save_best_only=False, period=epochs)
    tbCallback = callbacks.TensorBoard(log_dir=log_folder + data_name, histogram_freq=0,
                                       write_graph=True, update_freq='epoch')

    print(counter, ". Iteration der HPA\nlrate: ", lrate, "\nBatch Size: ", str(batch_size),
          "\nHidden Units: ", hidden_units, "\nLayers:", layer)

    print("Create architecture")
    input = k.layers.Input(shape=(data.input_train.shape[1],))
    if layer == 1:
        x = k.layers.Dense(units=hidden_units, activation='relu')(input)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=2, activation='linear')(x)
    elif layer == 2:
        x = k.layers.Dense(units=hidden_units, activation='relu')(input)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=2, activation='linear')(x)
    elif layer == 3:
        x = k.layers.Dense(units=hidden_units, activation='relu')(input)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=2, activation='linear')(x)
    elif layer == 4:
        x = k.layers.Dense(units=hidden_units, activation='relu')(input)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=2, activation='linear')(x)
    elif layer == 5:
        x = k.layers.Dense(units=hidden_units, activation='relu')(input)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=hidden_units, activation='relu')(x)
        x = k.layers.Dropout(rate=0.5)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dense(units=2, activation='linear')(x)

    optimizer = optimizers.adam(lr=lrate, beta_1=beta1, beta_2=beta2)
    model = k.Model(input, x)

    def mean_pred(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true - y_pred))

    def lat_loss(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))*data.stdv_output

    def lon_loss(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))*data.stdv_output

    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['accuracy', mean_pred, lat_loss, lon_loss])

    print("Start training")
    history = model.fit(x=np.array(data.input_train), y=np.array(data.output_train), batch_size=batch_size, epochs=epochs
                        , validation_data=(data.input_validation, data.output_validation), shuffle='batch',
                        callbacks=[stopCallback, checkpoint, tbCallback])
    name = str_learning_rate + "units_{}_layers_{}_batch_size{}_vall_loss{:.2f}" \
        .format(hidden_units, layer, batch_size, history.history['val_loss'][-1])
    save_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester"
                "\\Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\history\\" + folder_name + "\\"
                + name, history.history)

    counter += 1

    if history.history['val_loss'][-1] > 10:
        return 10

    return history.history['val_loss'][-1]


def fetch_bestes_ergebniss(path):
    """
        Searhces in the Bayes result object for the best Paramcombination
    :param path: Path to the bayes result file
    :return:
    """

    pickle_in = open(os.path.join("Ergebnisse", path, "Bayes_Pickle", "search_result_bayes_Ucdata.pickle"), 'rb')
    sr = pickle.load(pickle_in)

    # Ergebnisse der BayesOpt
    result = zip(sr.func_vals, sr.x_iters)
    sr_sorted = sorted(zip(sr.func_vals, sr.x_iters))

    print('Beste Hyperparameterkombination: ', sr.x)
    return sr.x[0], int(sr.x[1]), sr.x[2], sr.x[3]


if __name__ == "__main__":

    hoursToSleep = 0
    if hoursToSleep > 0:
        print("Sleeping...")
        time.sleep(60 * 60 * hoursToSleep)

    print("Running...")

    counter = 0
    result = gp_minimize(func=trainiere_knn,
                         dimensions=dimensions,
                         acq_func='EI',  # Expected Improvement
                         n_calls=no_bayes_iter,
                         x0=default_parameters)

    save_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\search_result_" + folder_name, result)


