import numpy as np
from math import exp


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, neuron_num_list, learning_rate):
        # basic neural net information
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._neuron_num_list = neuron_num_list
        self._learning_rate = learning_rate

        # construct network
        self._network = self._initialize_network()

    def train(self,x ,y):
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
                self._network['weight'][backward_step] = self._network['weight'][backward_step] + self._learning_rate
                self._network['bias'][backward_step] = self._network['bias'][backward_step] + self._learning_rate

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


if __name__ == '__main__':
    # forward propagation test
    nn = NeuralNetwork(5, 2, [3, 2, 2])
    vector = np.array([[1, 0, 1, 0, 1]])
    answer = np.array([[0.5, 0.5]])
    print(nn.predict(vector))
    print(nn.evaluate(vector, answer))
