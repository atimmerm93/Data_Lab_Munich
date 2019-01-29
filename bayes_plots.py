__author__ = "Alexander Timmermann"
__email__ = "alexandertimmermann93@gmail.com"
__status__ = "Production"

import sys
import os
sys.path.append("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                "Studienarbeit_Alexander_Timmermann\\Phyton\\GPS_KNN")
import numpy as np
import pickle
#from RNN_Bayes_Train import def_bayes_train
from Data_RNN import save_pickle, load_pickle
from matplotlib import pyplot
from skopt.plots import plot_convergence , partial_dependence, plot_evaluations, plot_objective
from personalisierte_skopt_plots import convergence_plot_mod, plot_evaluations_mod, plot_objective_mod
from GPS_ANN import trainiere_knn




class BayesResult():
    """
        Class for all required plots to validate the
        results of the bayes algorithm
    """

    def __init__(self, path, dimension_names):
        self.path = path
        self.results = load_pickle(path)
        self.dimension_names = dimension_names

    def return_best_func(self):
        min_loss = self.results.func_vals.min()
        iter_min_loss = sys.maxsize
        for i in range(len(self.results.func_vals)):
            if min_loss == self.results.func_vals[i]:
                iter_min_loss = i
                print("I ist: ", i)
        params_min_loss = self.results.x_iters[iter_min_loss]

        return params_min_loss, iter_min_loss, min_loss

    def plot_results_o_iters(self, history_path):
        """
            plots the training loss of each network and the validiton loss at the end
            of the training for each specific network

        :param history_path: path to history files -> needed for the training loss
        :return:
        """
        learning_rate = np.array(self.results.x_iters)[:, 1]
        results_o_iters = pyplot.figure()
        x = range(len(self.results.func_vals))
        pyplot.plot(x, self.results.func_vals, 'bo', markersize=15)
        history_files = os.listdir(history_path)
        last_training_loss = []
        for i in history_files:
            pickle_in = open(history_path + i, 'rb')
            history = pickle.load(pickle_in)
            last_training_loss.append(history['loss'][-1])
        """
        # Only for ppp_second_run_without_limits
            #0,3,4
        list_indexes = []
        list_indexes.append(1)
        list_indexes.append(2)
        for m in range(5, 100, 1):
            list_indexes.append(m)

        pyplot.plot(list_indexes, last_training_loss, 'ro', markersize=15)
        """
        pyplot.plot(range(len(last_training_loss)), last_training_loss, 'ro', markersize=15)
        pyplot.legend(["Valdierungs Loss", "Trainings Loss"], fontsize=15)
        pyplot.xlabel("Anzahl Iteration n", fontsize=40)
        pyplot.ylabel("Loss", fontsize=40)
        pyplot.grid("on")
        pyplot.xticks(size=20)
        pyplot.yticks(size=20)
        pyplot.show()
        results_copy = []
        iter_no = []
        for i, idx in zip(self.results.func_vals, range(len(self.results.func_vals))):
            if i < 3:
                results_copy.append(i)
                iter_no.append(idx)
        pyplot.figure()
        pyplot.plot(iter_no, results_copy, 'bo', markersize=15)
        pyplot.xlabel("Anzahl Iterationen n", fontsize=40)
        pyplot.ylabel("Validierungs Loss", fontsize=40)
        pyplot.grid("on")
        pyplot.xticks(size=20)
        pyplot.yticks(size=20)
        pyplot.show()

    def plot_range_best(self, min_range, max_range):
        """
            Shows best results in a certain range around the
            best result

        :param min_range: lowerbound of range
        :param max_range: upperbound of range
        """
        min_loss = self.results.func_vals.min()
        params_of_range_best_values = []
        range_best_value = []
        index_list = []
        for idx, val in enumerate(self.results.func_vals.tolist()):
            if min_loss - min_range < val < min_loss + max_range:
                params_of_range_best_values.append(self.results.x_iters[idx])
                print("Idx: ", idx)
                index_list.append(idx)
                range_best_value.append(val)
        pyplot.subplot(2, 1, 1)
        pyplot.plot(index_list, range_best_value)
        pyplot.grid()
        pyplot.subplot(2, 1, 2)
        pyplot.plot(range(len(range_best_value)), np.array(params_of_range_best_values)[:, 3])
        pyplot.grid()
        pyplot.show()
        print("Häufigkeit von 3-Layern: ", np.count_nonzero(np.array(params_of_range_best_values)[:, 3] == 3))

    def plot_partial_dependence(self):
        """plots parital dependence for the received results"""

        pax1, fig1 = plot_evaluations_mod(result=self.results, bins=len(self.dimension_names),
                                          dimension_names=self.dimension_names)

    def plot_objective(self):
        ax2, fig2 = plot_objective_mod(result=self.results, dimension_names=self.dimension_names)

    def plots(self, history_path):
        """generates all needed plots with one function"""
        self.plot_partial_dependence()
        self.plot_results_o_iters(history_path)
        self.plot_objective()
        newfig = pyplot.figure()
        convergence_plot_mod(self.results)

if __name__ == '__main__':

    print("System Path: ", sys.path)
    """ 
    m = BayesResult("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                    "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\"
                    "search_result",
                    ['batch_size', 'learning rate', 'units', 'layer'])
                    
    
    m = BayesResult("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                    "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\"
                    "search_result_MLP_replaced_by_ppp_data",
                    ["lrate", "beta1", "beta2", "batch_size", "hidden_units", "layer"])
    
    m = BayesResult("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                    "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\"
                    "search_result_MLP_null_replaced_missing",
                    ["lrate", "beta1", "beta2", "batch_size", "hidden_units", "layer"])
    
    #Without Limits File of MLP only
    m = BayesResult("C:\\Users\\Alexander_Timmermann\\Desktop\\"
                    "second_run_only_ppp_bayes_without_limit\\search_result_MLP_only_ppp",
                    ["lrate", "beta1", "beta2", "batch_size", "hidden_units", "layer"])
    """
    m = BayesResult("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\"
                    "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\bayes_result\\"
                    "search_result_MLP_only_ppp",
                    ["lrate", "beta1", "beta2", "batch_size", "hidden_units", "layer"])

    k = m.results.func_vals
    params_min_loss, iter_min_loss, min_loss = m.return_best_func()

    m.plots("C:\\Users\\Alexander_Timmermann\\PowerFolders\\3. Master Semester\\" \
       "Studienarbeit_Alexander_Timmermann\\Phyton\\Daten\\history\\MLP_only_ppp\\")
    #m.plot_results_o_iters("C:\\Users\\Alexander_Timmermann\\Desktop\\second_run_only_ppp_bayes_without_limit\\history\\")
    #convergence_plot_mod(m.results)
    print("Min Loss war: ", min_loss)




