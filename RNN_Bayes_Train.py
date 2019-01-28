from skopt.space import Real, Integer
from callbacks import StopTraining
from keras import callbacks
from keras.layers import *
from data_generator import load_pickle, save_pickle, data_generator
from skopt.utils import use_named_args
from skopt import gp_minimize
import tensorflow as tf
import os
from keras import optimizers
import keras as k


# 'Initialize params for bayes algorithm'
units = Integer(low=252, high=500, name='units')
learning_rate = Real(low=1e-12, high=5e-2, prior='log-uniform', name='learning_rate')
layers = Integer(low=1, high=3, name='layers')
batch_size = Integer(low=1, high=15, name='batch_size')
epochs = 30
counter = 0
no_bayes_iter = 50
default_parameters = [1, 1e-3, 260, 1]
input_path = 'C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a'
label_path = 'C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a'
dimensions = [learning_rate, units, layers, batch_size]


@use_named_args(dimensions=dimensions)
def def_bayes_train(learning_rate, units, layers, batch_size):
    """
        Traines a network with the parameters returned by the bayes algorithm
        :param input_path: path to input data
        :param label_path: path to label data
        :return:
    """
    # 'number of works dependence on number of cores of the system'
    n_workers = 16
    train_folder = input_path + "\\train\\input\\"
    input_shape = load_pickle(input_path + "\\train\\input\\traininput_0_16").shape
    output_shape = load_pickle(label_path + "\\train\\output\\trainoutput_0_16").shape
    model_folder = "Daten\\Modelle\\RNN_MLP\\"
    log_folder = "Daten\\Logs\\RNN_MLP\\"

    global counter

    print(counter, ". Iteration der HPA\nlrate: ", learning_rate, "\nBatch Size: ", str(batch_size*16)
              , "\nRecurrent Units: ", units, "\nLayers:", layers)

    print("Create generator")
    generatorTrain = data_generator(train=True, batch_size=batch_size)
    generatorValid = data_generator(valid=True, batch_size=batch_size)

    print("Create callbacks")
    stopCallback = StopTraining()
    str_learning_rate = "lrate_{:.3f}e-5".format(learning_rate * 100000)
    data_name = str_learning_rate + "Units_" + str(units) + "Layers_" + str(layers)
    checkpoint = callbacks.ModelCheckpoint(model_folder + data_name +
                                               "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',
                                               verbose=0, save_best_only=True, period=epochs)

    tbCallback = callbacks.TensorBoard(log_dir=log_folder + "\\" + data_name, histogram_freq=0, write_graph=True,
                                           update_freq='epoch')

    print("Create Graph of Network")
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    if layers == 1:
        x = CuDNNLSTM(units=units, return_sequences=True)(inputs)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = Dense(units=2, activation='linear')(x)
    elif layers == 2:
        x = CuDNNLSTM(units=units, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = CuDNNLSTM(units=units, return_sequences=True)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=2, activation='linear')(x)
    elif layers == 3:
        x = CuDNNLSTM(units=units, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = CuDNNLSTM(units=units, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = CuDNNLSTM(units=units, return_sequences=True)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units=2, activation='linear')(x)

    def mean_pred(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true - y_pred))

    def lat_loss(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

    def lon_loss(y_true, y_pred):
        return k.backend.mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

    optimizer = optimizers.rmsprop(lr=learning_rate, clipnorm=1)
    model = k.Model(inputs, x)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                      metrics=['accuracy', mean_pred, lat_loss, lon_loss])



    print("Start training")
    history = model.fit_generator(
        generator=generatorTrain,
        steps_per_epoch=
        int(len(os.listdir(input_path + "\\train\\output\\")))/5,
        epochs=epochs,
        verbose=1,
        validation_data=generatorValid,
        validation_steps=int(len(os.listdir(input_path + "\\train\\output\\")))/10,
        use_multiprocessing=False,
        workers=n_workers,
        max_queue_size=20,
        callbacks=[stopCallback, checkpoint, tbCallback])

    counter += 1

    print('Abgeschlossene Trainings: {}'.format(counter))
    print('Genauigkeit: {}\n '.format(history.history['val_loss'][-1]))

    # 'Create name for history file and save the history file'
    name = str_learning_rate + "units_{}_layers_{}_batch_size{}_vall_loss{:.2f}"\
            .format(units, layers, batch_size, history.history['val_loss'][-1])
    save_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester"
                    "\\Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\history\\RNN_MLP\\" + name, history.history)

    return history.history['val_loss'][-1]


def run_bayes_train(no_bayes_iter, default_parameters):
    """

    :param no_bayes_iter:
    :param default_parameters:
    :return:
    """
    search_result = gp_minimize(func=def_bayes_train,
                                dimensions=dimensions,
                                acq_func='EI',
                                n_calls=no_bayes_iter,
                                x0=default_parameters)
    save_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester"
                "\\Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\search_result_RNN_MLP",
                search_result)


if __name__ == '__main__':
    run_bayes_train(no_bayes_iter=no_bayes_iter, default_parameters=default_parameters)