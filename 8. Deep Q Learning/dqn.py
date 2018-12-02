import numpy as np
import random as rd
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization


class DeepQNet:
    def __init__(self, n_actions, n_features, learning_rate, momentum,
                 l2_penalty, fit_epoch, batch_size, discount_factor,
                 e_greedy, memory_size):
        # basic agent information
        self._n_actions = n_actions
        self._n_features = n_features
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._l2_penalty = l2_penalty
        self._fit_epoch = fit_epoch
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._e_greedy = e_greedy
        self._memory_size = memory_size

        # build neural network
        self._net = self._build_network()

        # memory pool
        self._memory_pool = []

    def train(self):
        # sample train data uniformly from memory pool to conduct experience replay
        sample_train_date = rd.sample(self._memory_pool, self._memory_size)
        observation_sample = np.array([sample[0] for sample in sample_train_date])

        # construct the target by Bellman Equation
        print(observation_sample.shape)
        target_q_value = self._net.predict(observation_sample)
        for index in range(len(target_q_value)):
            action = sample_train_date[index][1]
            reward = sample_train_date[index][2]
            observation_next = sample_train_date[index][3]
            is_done = sample_train_date[index][4]
            if not is_done:
                future_optimal_q = np.max(self._net.predict(observation_next))
            else:
                future_optimal_q = 0
            target_q_value[index][action] = reward + self._discount_factor * future_optimal_q

        # train the network
        self._net.fit(observation_sample, target_q_value, epochs=self._fit_epoch, batch_size=self._batch_size, verbose=1)

    def choose_action(self, observation):
        # use e-greedy strategy to generate action
        if rd.random() < self._e_greedy:
            # random action
            action = rd.randint(0, self._n_actions-1)
        else:
            # optimal action
            observation = np.reshape(observation, newshape=(1, 1, self._n_features))
            q_vector = self._net.predict(observation)
            action = np.argmax(q_vector)
        return action

    def store_train_data(self, observation_current, action, reward, observation_next, is_done):
        data_list = [observation_current, action, reward, observation_next, is_done]
        self._memory_pool.append(data_list)

    def have_enough_data(self):
        if len(self._memory_pool) >= self._memory_size:
            return True
        else:
            return False

    def _build_network(self):
        # define the input format
        init_x = Input((1, self._n_features))

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
