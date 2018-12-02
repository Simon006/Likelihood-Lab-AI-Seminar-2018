import numpy as np
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import add
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD


class DeepQNet:
    def __init__(self, n_actions, n_features, learning_rate, l2_coefficient, e_greedy, memory_size):
        # basic agent information
        self._n_actions = n_actions
        self._n_features = n_features
        self._learning_rate = learning_rate
        self._l2_coefficient = l2_coefficient
        self._e_greedy = e_greedy
        self._memory_size = memory_size

        # build neural network
        self._net = self._build_network()

    def train(self):
        pass

    def choose_action(self):
        pass

    def store_train_data(self):
        pass

    def have_enough_data(self):
        pass

    def _build_network(self):
        init_x = Input((1, 1, self._n_features))
        x = init_x

        x = Dense(15, kernel_regularizer=l2(self._l2_coefficient))(x)
