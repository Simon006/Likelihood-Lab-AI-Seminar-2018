import numpy as np
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD


class DeepQNet:
    def __init__(self, n_actions, n_features, learning_rate, momentum,
                 l2_penalty, discount_factor, e_greedy, memory_size):
        # basic agent information
        self._n_actions = n_actions
        self._n_features = n_features
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._l2_penalty = l2_penalty
        self._discount_factor = discount_factor
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
        # define the input format
        init_x = Input((1, 1, self._n_features))

        # Multiple Dense Layers
        x = Dense(15, kernel_regularizer=l2(self._l2_penalty))(init_x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(10, kernel_regularizer=l2(self._l2_penalty))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(self._n_actions, kernel_regularizer=l2(self._l2_penalty))(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        # define the network
        net = Model(inputs=init_x, outputs=x)

        # define the loss function
        opt = SGD(lr=self._learning_rate, momentum=self._momentum, nesterov=True)
        net.compile(optimizer=opt, loss='mean_squared_error')

        return net
