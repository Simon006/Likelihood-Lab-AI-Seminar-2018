"""
Likelihood Lab
XingYu
"""
from sklearn.datasets import load_breast_cancer
from math import exp
import matplotlib.pyplot as plt
import random as rd
import numpy as np


class Logistic:
    def __init__(self, input_size, learning_rate, epoch):
        # Hyper-parameter
        self._weight_num = input_size + 1  # Contains one bias term
        self._learning_rate = learning_rate
        self._epoch = epoch

        # Random Normal Distributed initial weight
        self._weight = np.random.randn(self._weight_num)

    def predict(self, x):
        y_predict = []
        for sample in x:
            result = self._logistic(np.dot(sample, self._weight[1:]) + self._weight[0])
            y_predict.append(result)

        y_predict = np.array(y_predict)
        return y_predict

    def train(self, x, y):
        # Check length error
        if len(x) != len(y):
            raise ValueError('The length of x and y do not match.')

        # Stochastic Gradient Descent
        for e in range(self._epoch):
            for i in range(len(x)):
                gradient = (self.predict(np.array([x[i]]))[0] - y[i]) * np.append(np.array([1]), x[i])
                self._weight = self._weight - self._learning_rate * gradient

    def evaluate(self, x, y):
        # Check length error
        if len(x) != len(y):
            raise ValueError('The length of x and y do not match.')

        # Prediction
        y_prediction = self.predict(x)

        # MSE
        difference = y_prediction - y
        mse = sum([diff**2 for diff in difference]) / len(difference)

        # Accuracy
        y_prediction_class = np.round(y_prediction)
        correct_num = 0
        for i in range(len(y_prediction)):
            if y[i] == y_prediction_class[i]:
                correct_num += 1
            else:
                continue
        accuracy = correct_num / len(y)
        return mse, accuracy

    def feature_importance(self):
        pass

    def _logistic(self, z):
        return 1 / (1 + exp(-z))


if __name__ == '__main__':
    # Import iris data
    breast_cancer = load_breast_cancer()

    # Separate input(x) and output(y)
    data_x = breast_cancer['data']
    data_y = breast_cancer['target']

    # Normalize the data's column
    for j in range(len(data_x[0])):
        min_value = min([data_x[i][j] for i in range(len(data_x))])
        max_value = max([data_x[i][j] for i in range(len(data_x))])
        for i in range(len(data_x)):
            data_x[i][j] = (data_x[i][j] - min_value) / (max_value - min_value)

    # Shuffle the data
    random_idx = rd.sample([i for i in range(len(data_x))], len(data_x))
    data_x = data_x[random_idx]
    data_y = data_y[random_idx]

    # Separate training and testing data set
    train_rate = 0.1
    sample_num = len(data_x)
    train_sample_num = int(train_rate * sample_num)
    train_x = data_x[:train_sample_num]
    train_y = data_y[:train_sample_num]
    test_x = data_x[train_sample_num:]
    test_y = data_y[train_sample_num:]

    lg = Logistic(len(data_x[0]), 0.1, 400)
    lg.train(train_x, train_y)
    mse, acc = lg.evaluate(test_x, test_y)
    print(mse)
    print(acc)
