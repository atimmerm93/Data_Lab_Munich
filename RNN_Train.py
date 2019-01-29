__author__ = "Alexander Timmermann"
__email__ = "alexandertimmermann93@gmail.com"
__status__ = "Done"

import os.path
import sys
sys.path.append("../")
import time
import keras as k
from keras import optimizers
import tensorflow as tf
from callbacks import StopTraining
from keras import callbacks
from keras.layers import *
from data_generator import load_pickle, save_pickle, data_generator


class TrainRNN:
    """
        Class to train the neural network for given specific parameters
    """

    def __init__(self, units, learning_rate, layers, batch_size, epochs):
        """

        :param units: specific value
        :param learning_rate: specific value
        :param layers: specific value
        :param batch_size: specific value
        :param epochs: specific value
        :return:
        """
        self.units = units
        self.learning_rate = learning_rate
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def network_train(self):
        """
        Trains the network for specific parameters

        :return:
        """
        epochs = 5000
        input_path = "C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a"
        output_path = "C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a"

        train_folder = input_path + "\\train\\input\\"
        input_shape = \
            load_pickle(input_path + "\\train\\input\\traininput_0_16").shape
        output_shape = \
            load_pickle(output_path + "\\train\\output\\trainoutput_0_16").shape

        model_folder = "Daten\\Modelle\\RNN_RNN_MLP\\"
        log_folder = "Daten\\Logs\\RNN_RNN_MLP\\"
        global counter

        print(counter, ". Iteration der HPA\nlrate: ", self.learning_rate, "\nBatch Size: ", str(self.batch_size * 16)
              , "\nRecurrent Units: ", self.units, "\nLayers:", self.layers)

        print("Create generator")
        generatorTrain = data_generator(train=True, batch_size=self.batch_size)
        generatorValid = data_generator(valid=True, batch_size=self.batch_size)

        print("Create callbacks")
        stopCallback = StopTraining()
        str_learning_rate = "lrate_{:.3f}e-5".format(self.learning_rate * 100000)
        data_name = str_learning_rate + "Units_" + str(self.units) + "Layers_" + str(self.layers)
        checkpoint = callbacks.ModelCheckpoint(model_folder + data_name +
                                               "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',
                                               verbose=0, save_best_only=False, period=30)
        tbCallback = callbacks.TensorBoard(log_dir=log_folder + "\\" + data_name, histogram_freq=0, write_graph=True,
                                           update_freq='epoch')
        reduce_learning_rate = callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-12)

        inputs = Input(shape=(input_shape[1], input_shape[2]))

        if self.layers == 1:
            x = CuDNNLSTM(units=self.units, return_sequences=False)(inputs)
            x = Dense(units=2, activation='linear')(x)

        elif self.layers == 2:
            x = CuDNNLSTM(units=self.units, return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = CuDNNLSTM(units=self.units, return_sequences=False)(x)
            x = Dense(units=2, activation='linear')(x)

        elif self.layers == 3:
            x = CuDNNLSTM(units=self.units, return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = CuDNNLSTM(units=self.units, return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = CuDNNLSTM(units=self.units, return_sequences=False)(x)
            x = Dense(units=2, activation='linear')(x)

        def mean_pred(y_true, y_pred):
            return k.backend.mean(tf.abs(y_true - y_pred))

        def lat_loss(y_true, y_pred):
            return k.backend.mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

        def lon_loss(y_true, y_pred):
            return k.backend.mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

        optimizer = optimizers.rmsprop(lr=self.learning_rate, clipnorm=1)
        model = k.Model(inputs, x)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', mean_pred, lat_loss, lon_loss])

        n_workers = 16
        print("Start training")
        history = model.fit_generator(
            generator=generatorTrain,
            steps_per_epoch=
            int(len(os.listdir(input_path + "\\train\\output\\"))) / 5,
            epochs=epochs,
            verbose=1,
            validation_data=generatorValid,
            validation_steps=int(len(os.listdir(input_path + "\\train\\output\\"))) / 10,
            use_multiprocessing=False,
            workers=n_workers,
            max_queue_size=20,
            callbacks=[stopCallback, checkpoint, tbCallback, reduce_learning_rate])

        counter += 1

        print('Abgeschlossene Trainings: {}'.format(counter))
        print('Genauigkeit: {}\n '.format(history.history['val_loss'][-1]))

        # 'Create name for history file and save the history file'
        name = str_learning_rate + "units_{}_layers_{}_batch_size{}_vall_loss{:.2f}" \
            .format(self.units, self.layers, self.batch_size, history.history['val_loss'][-1])
        save_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester"
                    "\\Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\history\\" + name, history.history)


if __name__ == '__main__':
    """
        If Training should start after a certain amount of hours 
        the hoursToSleep Variable has to be set to that specific 
        value
    """

    print("Sleeping...")
    hoursToSleep = 0

    if hoursToSleep > 0:
        print("Sleeping...")
        time.sleep(60 * 60 * hoursToSleep)

    """
        Initialize the parameters of the bayes algorithm
    """
    print("Running...")


    """
    m = load_pickle("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester"
                "\\Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\search_result")
    rnn = TrainRNN(m.x_iters[27][0], m.x_iters[27][1], m.x_iters[27][2], m.x_iters[27][3])
    network_train()
    """