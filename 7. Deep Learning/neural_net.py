import numpy as np
from math import exp
import random as rd
from sklearn.datasets import load_wine


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, neuron_num_list, learning_rate, epoch):
        # basic neural net information
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._neuron_num_list = neuron_num_list
        self._learning_rate = learning_rate
        self._epoch = epoch

        # construct network
        self._network = self._initialize_network()

    def train(self,x ,y):
        for e in range(self._epoch):
            # minimize the loss function in each training sample through SGD
            for i in range(len(x)):
                input_output_record = []
                tensor = x[i]
                # forward propagation
                for forward_step in range(len(self._network)):
                    input_output_record[forward_step] = dict()
                    input_output_record[forward_step]['input'] = tensor
                    tensor = np.dot(self._network[i]['weight'], tensor) + self._network[i]['bias']
                    tensor = _sigmoid_activation(tensor)
                    input_output_record[forward_step]['output'] = tensor

                # backward propagation
                for backward_step in reversed(range(len(self._network)-1)):
                    self._network['weight'][backward_step] = self._network['weight'][backward_step] \
                                                             + self._learning_rate * \
                                                             input_output_record[backward_step]['input']
                    self._network['bias'][backward_step] = self._network['bias'][backward_step] \
                                                           + self._learning_rate * \
                                                           np.ones(len(self._network['bias'][backward_step]))

    def predict(self, x):  # forward propagation
        y_predict = np.zeros((len(x), self._output_dim))
        for index, sample in enumerate(x):
            tensor = sample
            for layer in self._network:
                tensor = np.dot(layer['weight'], tensor) + layer['bias']
                tensor = _sigmoid_activation(tensor)
            y_predict[index] = tensor
        return y_predict

    def evaluate(self, x, y):
        y_predict = self.predict(x)
        y_difference = y_predict - y
        mse = np.average([np.dot(row, row) for row in y_difference])
        return mse

    def _initialize_network(self):
        # check mistake
        if self._output_dim != self._neuron_num_list[-1]:
            raise ValueError('output dimensionality does not match the neuron number of the last layer.')

        network = []
        for index, neuron_num in enumerate(self._neuron_num_list):
            layer = dict()
            if index == 0:
                layer['weight'] = np.random.normal(loc=0,
                                                   scale=0.01,
                                                   size=(neuron_num, self._input_dim))
                layer['bias'] = np.random.normal(loc=0,
                                                 scale=0.01,
                                                 size=neuron_num)
            else:
                layer['weight'] = np.random.normal(loc=0,
                                                   scale=0.01,
                                                   size=(neuron_num, self._neuron_num_list[index-1]))
                layer['bias'] = np.random.normal(loc=0,
                                                 scale=0.01,
                                                 size=neuron_num)
            network.append(layer)

        return network


def _sigmoid_activation(vector):
    result = np.zeros(len(vector))
    for index, value in enumerate(vector):
        result[index] = 1 / (1 + exp(-value))
    return result


def _derivative_sigmoid_activation(vector):
    return _sigmoid_activation(vector) - _sigmoid_activation(vector) * _sigmoid_activation(vector)


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
                        neuron_num_list=[15, 10, 5, len(train_y[0])],
                        learning_rate=0.1, epoch=20)
    dnn.train(x=train_x, y=train_y)
    mse = dnn.evaluate(x=test_x, y=test_y)
    print('Test MSE: ' + str(mse))
