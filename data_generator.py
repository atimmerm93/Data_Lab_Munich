__author__ = "Alexander Timmermann"
__email__ = "alexandertimmermann93@gmail.com"
__status__ = "Done"

import threading
import os
import pickle
import numpy as np

def load_pickle(path):
    pickle_in = open(path, 'rb')
    variable_to_load = pickle.load(pickle_in)
    pickle_in.close()
    return variable_to_load


def save_pickle(path, variable_to_save):
    pickle_out = open(path, 'wb')
    pickle.dump(variable_to_save, pickle_out)
    pickle_out.close()


class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

  def __next__(self):
      with self.lock:
          return self.it.__next__()

def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g


@threadsafe_generator
def data_generator(batch_size, train=False, valid=False, test=False):
    """

    :param batch_size: Number of samples to return
    :param train: wether to use the train
    :param valid:  or the valid
    :param test: or the test data
    :return: yields the batch and the corresponding label
    """
    path = ""
    if train:
        path = "train"
    elif valid:
        path = "validation"
    input_path = "C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a\\" + path + "\\input\\"
    output_path = "C:\\Users\\Alexander_Timmermann\\Desktop\\Studienarbeit\\ppp_a\\" + path + "\\output\\"
    list_of_input = os.listdir(input_path)
    list_of_output = os.listdir(output_path)

    i = 0
    while True:

        if i < len(list_of_input) - 1:
            i += 1
        else:
            i = 0
        random_choice = np.random.choice(len(list_of_input), batch_size, replace=False)
        for k in range(0, batch_size, 1):
            if k == 0:
                x = load_pickle(input_path + list_of_input[random_choice[k]])
                y = load_pickle(output_path + list_of_output[random_choice[k]])
            else:
                x = np.concatenate((x, load_pickle(input_path + list_of_input[random_choice[k]])), axis=0)
                y = np.concatenate((y, load_pickle(output_path + list_of_output[random_choice[k]])), axis=0)

        yield x, y