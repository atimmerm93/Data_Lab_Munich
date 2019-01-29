__author__ = "Alexander Timmermann"
__email__ = "alexandertimmermann93@gmail.com"
__status__ = "Done"

######################################################
##### Skript für alle standardmäßigen Preprocessing
##### Schritte (Split und Normalisierung)
##### wird alles in dem data preprocess Objekt hinterlegt
######################################################

import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class data_preprocess():
    """
        preprocess and normalizes the data for the feedforward training
    """
    def __init__(self, path, mit_pseudoranges):
        """
            stores all relevant training data and the related informations -> like
            the mean and the variance if the data is normalised

        :param path: path to the Data
        :param mit_pseudoranges: if is data with pseudoranges or without pseudranges
        """
        self.path = path
        self.mit_pseudoranges = mit_pseudoranges
        self.input_train = pd.DataFrame()
        self.input_validation = pd.DataFrame()
        self.input_test = pd.DataFrame()
        self.output_validation = pd.DataFrame()
        self.output_test = pd.DataFrame()
        self.output_train = pd.DataFrame()
        self.stdv_output = None
        self.stdv_input = None
        self.varianz_input = None
        self.varianz_output = None
        self.mittelwert_input = None
        self.mittelwert_output = None
        self.splitdata()

    def importdata(self):
        """
            Loads the data from hard drive
        :param fahrt_no: number of the specific ride
        :param buchstabe: wether the ride was recoreded in ppp or normal mode
        :return:
        """
        if self.mit_pseudoranges:
            df = pd.read_csv(self.path, index_col=0, dtype=np.float64)
        else:
            df = pd.read_csv(os.path.join("Daten", "Messdaten", "kompletter_Datensatz.csv"), index_col=0, dtype=np.float64)
        print("Shape der Daten: ", df.shape)
        return df

    def splitdata(self):
        """
            Splits the data in Test Train and Valid -> 80% Test ,10% Valid and 10% Test
            and rearranges the data randomly
        :return:
        """

        df = self.importdata()

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_abweichungbool = df[(df[['Abweichung']] > 0.05).all(axis=1)]

        df.drop(df.index[df_abweichungbool.index], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df.head())

        df.drop(['Abweichung', 'UTC Zeit', 'TimeFromStart (s)', 'Time (UTC+2,00)'], axis=1, inplace=True)
        print(df.shape)
        input = df.drop(['Lat', 'Lon', 'Abweichung Lon', 'Abweichung Lat'], axis=1)
        output = df.loc[:, ['Abweichung Lat', 'Abweichung Lon']]
        input.reset_index(inplace=True, drop=True)
        output.reset_index(inplace=True, drop=True)

        input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(input, output,
                                                                                                      test_size=0.2,
                                                                                                      random_state=42)

        input_validation,  input_test, output_validation, output_test = sklearn.\
            model_selection.train_test_split(input_test, output_test, test_size=0.5, random_state=42)

        self.input_train = input_train
        self.input_validation = input_validation
        self.input_test = input_test

        self.output_train = output_train
        self.output_validation = output_validation
        self.output_test = output_test

    def normalization_of_data(self):
        self.splitdata()


        # Z-Normalisierung der Daten
        scaler = StandardScaler()
        scaler2 = MinMaxScaler()

        scaler.fit(self.input_train[self.input_train.columns])
        self.input_train[self.input_train.columns] = scaler.transform(self.input_train[self.input_train.columns])
        self.input_test[self.input_test.columns] = scaler.transform(self.input_test[self.input_test.columns])
        self.input_validation[self.input_validation.columns] = scaler.transform(self.input_validation[self.input_validation.columns])

        self.mittelwert_input = scaler.mean_
        self.varianz_input = scaler.var_
        self.stdv_input = np.sqrt(self.varianz_input)

        scaler2.fit(self.output_train[self.output_train.columns])
        scaler.fit(self.output_train[self.output_train.columns])
        self.output_train[self.output_train.columns] = scaler.transform(self.output_train[self.output_train.columns])
        self.output_test[self.output_test.columns] = scaler.transform(self.output_test[self.output_test.columns])
        self.output_validation[self.output_validation.columns] = scaler.transform(self.output_validation[self.output_validation.columns])

        self.mittelwert_output = scaler.mean_
        self.varianz_output = scaler.var_
        self.stdv_output = np.sqrt(self.varianz_output)
        min = scaler2.data_min_
        max = scaler2.data_max_
        dif = max - min
        k = self.input_test.columns.get_loc("SV G9 Az (L1C/A)")
        #Beispielhafter Wert um Standard Scaler zu prüfen rückrechnung ergibt genau -8.10 also korrekt
        wert38575_nach_scaler = self.output_train[self.output_train.index == 38576]

