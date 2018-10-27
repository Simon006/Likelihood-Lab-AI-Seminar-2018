"""
Likelihood Lab
XingYu
"""
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
import random as rd


class Node:
    def __init__(self):
        self.split_index = None
        self.split_value = None
        self.leaf_node = None
        self.right_node = None
        self.is_terminal = False
        self.category = None
        self.depth = 0


class DecisionTree:
    def __init__(self, input_dim, class_num, maximal_depth, minimal_samples):
        # Basic Classifier Information
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
                    current_node = current_node.leaf_node
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
        condition1 = len(x) > self._minimal_samples
        condition2 = node.depth < self._maximal_depth

        if condition1 and condition2:
            x_left, y_left, x_right, y_right = self._split(node, x, y)
            if len(x_left) == 0 or len(x_right) == 0:
                node.is_terminal = True
                node.category = max(list(y), key=list(y).count)
            else:
                node_left = Node()
                node_left.depth = node.depth + 1
                node.leaf_node = self._build_tree(node_left, x_left, y_left)
                node_right = Node()
                node_right.depth = node.depth + 1
                node.right_node = self._build_tree(node_right, x_right, y_right)
        else:
            node.is_terminal = True
            node.category = max(list(y), key=list(y).count)
            print(node.category)
            return node

        return node

    def _split(self, node, x, y):

        best_value = 10000000000000
        best_index = None
        best_x_left = None
        best_y_left = None
        best_x_right = None
        best_y_right = None

        for i in range(self._input_dim):
            for sample in x:

                gini_value, x_left, y_left, x_right, y_right = self._gini(i, sample[i], x, y)

                if gini_value < best_value:
                    best_value = gini_value
                    best_index = i
                    best_x_left = x_left
                    best_y_left = y_left
                    best_x_right = x_right
                    best_y_right = y_right

        node.split_index = best_index
        node.split_value = best_value
        return best_x_left, best_y_left, best_x_right, best_y_right

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
        gini_value_left = 1 - sum([(left_y_stat_dict[key]/len(left_y_list))**2 for key in left_y_stat_dict])
        right_y_stat_dict = Counter(right_y_list)
        gini_value_right = 1 - sum([(right_y_stat_dict[key]/len(right_y_list))**2 for key in right_y_stat_dict])
        gini_value = gini_value_left + gini_value_right

        return gini_value, np.array(left_x_list), np.array(left_y_list), np.array(right_x_list), np.array(right_y_list)


if __name__ == '__main__':

    """
    still testing
    """

    iris = load_iris()

    iris_x = iris['data']
    iris_y = iris['target']

    # Shuffle the data
    random_idx = rd.sample([i for i in range(len(iris_x))], len(iris_x))
    iris_x = iris_x[random_idx]
    iris_y = iris_y[random_idx]

    dt = DecisionTree(4, 3, 10000000000, 2)
    dt.train(iris_x, iris_y)
    acc = dt.evaluate(iris_x, iris_y)
    print(acc)
    print(dt.predict(iris_x))
