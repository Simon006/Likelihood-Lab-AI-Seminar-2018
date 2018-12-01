import numpy as np
from math import exp
import random as rd
from sklearn.datasets import load_wine


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, neuron_list, activation_list, learning_rate, epoch):
        # basic neural net information
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._neuron_list = neuron_list
        self._activation_list = activation_list
        self._learning_rate = learning_rate
        self._epoch = epoch

        # construct network
        self._network = self._initialize_network()

    def train(self,x ,y):
        for e in range(self._epoch):
            square_error_sum = 0
            # minimize the loss function in each training sample by gradient descend
            # the loss function is defined as the
            for index, tensor in enumerate(x):
                input_output_record = []
                # forward propagation
                for forward_step in range(len(self._network)):
                    input_output_record[forward_step] = dict()

                    # conduct linear transformation and record the input and linear output
                    input_output_record[forward_step]['input'] = tensor
                    tensor = np.dot(self._network[forward_step]['weight'], tensor) + self._network[forward_step]['bias']
                    input_output_record[forward_step]['linear_output'] = tensor

                    # non-linear transformation
                    if self._network[forward_step]['activation'] == 'relu':
                        tensor = _rectified_linear_unit(tensor)
                    elif self._network[forward_step]['activation'] == 'sigmoid':
                        tensor = _sigmoid(tensor)
                    else:
                        raise ValueError

                # add Square error in this sample
                square_error_sum += sum(np.dot(tensor - y[index], tensor - y[index]))

                # backward propagation
                for backward_step in reversed(range(len(self._network)-1)):
                    self._network['weight'][backward_step] = self._network['weight'][backward_step] \
                                                             + self._learning_rate * \
                                                             input_output_record[backward_step]['input']
                    self._network['bias'][backward_step] = self._network['bias'][backward_step] \
                                                           + self._learning_rate * \
                                                           np.ones(len(self._network['bias'][backward_step]))

            # print the error in this training epoch
            mse = square_error_sum / len(x)
            print('------The MSE of the {0} epoch is {1}------'.format(str(e + 1), mse))

    def predict(self, x):
        y_predict = np.zeros((len(x), self._output_dim))
        for index, sample in enumerate(x):
            tensor = sample

            # forward propagation
            for layer in self._network:
                # linear transformation
                tensor = np.dot(layer['weight'], tensor) + layer['bias']

                # non-linear transformation
                if layer['activation'] == 'relu':
                    tensor = _rectified_linear_unit(tensor)
                elif layer['activation'] == 'sigmoid':
                    tensor = _sigmoid(tensor)
                else:
                    raise ValueError

            y_predict[index] = tensor
        return y_predict

    def evaluate(self, x, y):
        y_predict = self.predict(x)
        y_difference = y_predict - y
        mean_square_error = np.average([np.dot(row, row) for row in y_difference])
        return mean_square_error

    def _initialize_network(self):
        # check mistake
        if self._output_dim != self._neuron_list[-1]:
            raise ValueError('output dimensionality does not match the neuron number of the last layer.')

        if len(self._neuron_list) < len(self._activation_list):
            raise ValueError('each layer can only have one activation function.')
        elif len(self._neuron_list) > len(self._activation_list):
            raise ValueError('every layer must be equipped with one activation function.')
        else:
            pass

        # initialize the network's layer
        network = []
        for index, neuron_num in enumerate(self._neuron_list):
            layer = dict
            # define layer weight and bias
            if index == 0:
                layer['weight'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num, self._input_dim + 1))
            else:
                layer['weight'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num, self._neuron_list[index-1] + 1))
            layer['bias'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num, 1))

            # define the activation function(you have two options: rectified linear unit or sigmoid)
            if self._activation_list[index] not in {'relu', 'sigmoid'}:
                raise ValueError('activation not available.')
            else:
                layer['activation'] = self._activation_list[index]

            network.append(layer)

        return network


def _sigmoid(vector):
    result = np.zeros(len(vector))
    for index, value in enumerate(vector):
        result[index] = 1 / (1 + exp(-value))
    return result


def _derivative_sigmoid(vector):
    return _sigmoid(vector) - _sigmoid(vector) * _sigmoid(vector)


def _rectified_linear_unit(vector):
    vector[vector < 0] = 0
    return vector


def _derivative_rectified_linear_unit(vector):
    vector[vector < 0] = 0
    vector[vector > 0] = 1
    return vector


def num_2_one_hot(y):
    one_hot_array = np.zeros((len(y), len(set(y))))
    response_2_category = dict()
    for index, response in enumerate(set(y)):
        response_2_category[response] = index
    for index, response in enumerate(y):
        one_hot_array[index][response_2_category[response]] = 1
    return one_hot_array


if __name__ == '__main__':
    # load wine data
    # each sample has 13 features and 3 possible classes
    wine = load_wine()
    wine_x = wine['data']
    wine_y = wine['target']

    # shuffle the data randomly
    random_idx = rd.sample([i for i in range(len(wine_x))], len(wine_x))
    wine_x = wine_x[random_idx]
    wine_y = wine_y[random_idx]

    # convert label into one hot format
    wine_y = num_2_one_hot(wine_y)

    # split the data into training data set and testing data set
    train_rate = 0.7
    train_num = int(train_rate*len(wine_x))
    train_x = wine_x[:train_num]
    train_y = wine_y[:train_num]
    test_x = wine_x[train_num:]
    test_y = wine_y[train_num:]

    # train neural net to predict
    dnn = NeuralNetwork(input_dim=len(train_x[0]), output_dim=len(train_y[0]),
                        neuron_list=[15, 10, 5, len(train_y[0])], activation_list=['relu', 'relu', 'relu', 'sigmoid'],
                        learning_rate=0.1, epoch=20)
    dnn.train(x=train_x, y=train_y)
    mse = dnn.evaluate(x=test_x, y=test_y)
    print('Test MSE: ' + str(mse))
