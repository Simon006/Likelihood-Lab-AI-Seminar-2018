"""
Likelihood Lab
XingYu
"""


import numpy as np
import random as rd
from collections import Counter
from sklearn.datasets import load_iris


class Node:
    def __init__(self):
        # node information
        self.depth = 0
        self.split_index = None
        self.split_value = None
        self.category = None
        self.is_terminal = False

        # children nodes
        self.left_node = None
        self.right_node = None


class DecisionTree:
    def __init__(self, input_dim, class_num, maximal_depth, minimal_samples):
        # basic classifier information
        self._input_dim = input_dim
        self._class_num = class_num
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples

        # Define the tree root
        self._root = Node()

    def train(self, x, y):
        self._build_tree(self._root, x, y)

    def predict(self, x):
        y_predict = []
        for sample in x:

            current_node = self._root

            while True:
                if current_node.is_terminal:
                    y_predict.append(current_node.category)
                    break
                elif sample[current_node.split_index] < current_node.split_value:
                    current_node = current_node.left_node
                else:
                    current_node = current_node.right_node

        return np.array(y_predict)

    def evaluate(self, x, y):

        y_predict = self.predict(x)

        correct_num = 0
        for i in range(len(y)):
            if y[i] == y_predict[i]:
                correct_num += 1
            else:
                continue

        accuracy = correct_num / len(y)

        return accuracy

    def _build_tree(self, node, x, y):
        condition1 = len(x) > self._minimal_samples  # enough training data
        condition2 = node.depth < self._maximal_depth  # avoid tree that is too complicated

        if condition1 and condition2:
            best_split_index, best_split_value, x_left, y_left, x_right, y_right = self._split(x, y)
            if (len(x_left) == 0) or (len(x_right) == 0):
                node.split_index = None
                node.split_value = None
                node.is_terminal = True
                node.category = max(list(y), key=list(y).count)
                node.left_node = None
                node.right_node = None
                return 0
            else:
                node.split_index = best_split_index
                node.split_value = best_split_value
                node.is_terminal = False
                node.category = None
                node.left_node = Node()
                node.right_node = Node()
                node.left_node.depth = node.depth + 1
                node.right_node.depth = node.depth + 1
                self._build_tree(node.left_node, x_left, y_left)
                self._build_tree(node.right_node, x_right, y_right)
        else:
            node.split_index = None
            node.split_value = None
            node.is_terminal = True  # create leaf node
            node.category = max(list(y), key=list(y).count)  # majority voting
            node.left_node = None
            node.right_node = None
            return 0

    def _split(self, x, y):

        best_gini = 100000000
        best_split_index = None
        best_split_value = None
        best_x_left = None
        best_y_left = None
        best_x_right = None
        best_y_right = None

        for i in range(self._input_dim):
            for sample in x:
                gini, x_left, y_left, x_right, y_right = self._gini(i, sample[i], x, y)

                if gini < best_gini:
                    best_gini = gini
                    best_split_index = i
                    best_split_value = sample[i]
                    best_x_left = x_left
                    best_y_left = y_left
                    best_x_right = x_right
                    best_y_right = y_right

        return best_split_index, best_split_value, best_x_left, best_y_left, best_x_right, best_y_right

    def _gini(self, index, value, x, y):
        left_x_list = []
        left_y_list = []
        right_x_list = []
        right_y_list = []

        for i in range(len(x)):
            sample = x[i]
            label = y[i]
            if sample[index] < value:
                left_x_list.append(sample)
                left_y_list.append(label)
            else:
                right_x_list.append(sample)
                right_y_list.append(label)

        left_y_stat_dict = Counter(left_y_list)
        gini_value_left = sum([(left_y_stat_dict[key]/len(left_y_list))*(1-(left_y_stat_dict[key]/len(left_y_list))) for key in left_y_stat_dict])

        right_y_stat_dict = Counter(right_y_list)
        gini_value_right = sum([(right_y_stat_dict[key]/len(right_y_list))*(1-(right_y_stat_dict[key]/len(right_y_list))) for key in right_y_stat_dict])

        gini_value = gini_value_left + gini_value_right

        return gini_value, np.array(left_x_list), np.array(left_y_list), np.array(right_x_list), np.array(right_y_list)


if __name__ == '__main__':

    iris = load_iris()

    iris_x = iris['data']
    iris_y = iris['target']

    # Shuffle the data
    random_idx = rd.sample([i for i in range(len(iris_x))], len(iris_x))
    iris_x = iris_x[random_idx]
    iris_y = iris_y[random_idx]

    train_x = iris_x[:int(0.5*len(iris_x))]
    train_y = iris_y[:int(0.5*len(iris_y))]
    test_x = iris_x[int(0.5*len(iris_x)):]
    test_y = iris_y[int(0.5*len(iris_y)):]

    dt = DecisionTree(len(iris_x[0]), 3, 15, 2)
    dt.train(train_x, train_y)
    acc = dt.evaluate(test_x, test_y)
    print(acc)
