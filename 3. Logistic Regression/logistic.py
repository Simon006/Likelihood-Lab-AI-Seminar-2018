"""
Likelihood Lab
XingYu
"""
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import random as rd


class Logistic:
    def __init__(self, input_size, learning_rate, batch_size, epoch):
        self._input_size = input_size
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epoch = epoch

    def train(self, x, y):
        pass

    def predict(self, x, y):
        pass


if __name__ == '__main__':
    # Import iris data
    iris = load_breast_cancer()

    # Separate input(x) and output(y)
    data_x = iris['data']
    data_y = iris['target']

    # Shuffle the data
    random_idx = rd.sample([i for i in range(len(data_x))], len(data_x))
    data_x = data_x[random_idx]
    data_y = data_y[random_idx]

    # Separate training and testing data set
    train_rate = 0.7
    sample_num = len(data_x)
    train_sample_num = int(train_rate * sample_num)
    train_x = data_x[:sample_num]
    train_y = data_y[:sample_num]
    test_x = data_x[sample_num:]
    test_y = data_y[sample_num:]

    # Construct Logistic Predictive Model
    logistic = Logistic()

    # Train the algorithm
    logistic.train(train_x, train_y)

    # Test the algorithm
    logistic.predict(test_x, test_y)
