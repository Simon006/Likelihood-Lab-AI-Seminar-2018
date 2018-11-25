import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, neuron_num_list):
        # basic neural net information
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._neuron_num_list = neuron_num_list

        # construct network
        self._network = self._initialize_network()

    def train(self,x ,y):
        pass

    def predict(self, x):
        y_predict = np.zeros((len(x), self._output_dim))
        for index, sample in enumerate(x):
            temp = sample
            for layer in self._network:
                temp = np.dot(layer['weight'], temp) + layer['bias']
                temp = _sigmoid_activation(temp)
            y_predict[index] = temp
        return y_predict

    def evaluate(self, x, y):
        y_predict = self.predict(x)
        return

    def _initialize_network(self):
        # check mistake
        if self._output_dim != self._neuron_num_list[-1]:
            raise ValueError('output dimensionality does not match the neuron number of the last layer.')

        network = []
        for index, neuron_num in enumerate(self._neuron_num_list):
            layer = dict()
            if index == 0:
                layer['weight'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num, self._input_dim))
                layer['bias'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num))
            else:
                layer['weight'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num, self._neuron_num_list[index-1]))
                layer['bias'] = np.random.normal(loc=0, scale=0.01, size=(neuron_num))

            network.append(layer)

        return network


def _sigmoid_activation(x):

    return
