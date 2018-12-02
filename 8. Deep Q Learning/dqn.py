import numpy as np
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization


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

        # memory pool
        self._memory_pool = []

    def train(self):
        pass

    def choose_action(self, observation):
        # use e-greedy strategy to generate action
        if np.random.random() < self._e_greedy:
            # random action
            action = np.random.randint(0, self._n_actions)
        else:
            # optimal action
            q_vector = self._net.predict(observation)
            action = np.argmax(q_vector)

        return action

    def store_train_data(self, observation_current, action, reward, observation_next):
        data_tuple = (observation_current, action, reward, observation_next)
        self._memory_pool.append(data_tuple)

    def have_enough_data(self):
        if len(self._memory_pool) >= self._memory_size:
            return True
        else:
            return False

    def _build_network(self):
        # define the input format
        init_x = Input((1, 1, self._n_features))

        # multiple dense layers
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
