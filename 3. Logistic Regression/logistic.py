"""
Likelihood Lab
XingYu
"""
from sklearn.datasets import load_breast_cancer
import random as rd

class Logistic:
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate


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
    print(data_y)

    # Separate training and testing data set
