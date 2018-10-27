"""
Likelihood Lab
XingYu
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Node:
    def __init__(self):
        self.split_index = None
        self.split_value = None
        self.leaf_node = None
        self.right_node = None
        self.is_terminal = None
        self.category = None


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
        pass

    def evaluate(self, x, y):
        pass

    def _built_tree(self, node, x, y):
        node_left, x_left, y_left, node_right, x_right, y_right = self._split(node, x, y)
        self._built_tree(node_left, x_left, y_left)
        self._built_tree(node_right, x_right, y_right)

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
        node.leaf_node = Node()
        node.right_node = Node()
        return node.leaf_node, best_x_left, best_y_left, node.right_node, best_x_right, best_y_right

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

        gini_value =

        return gini_value, np.array(left_x_list), np.array(left_y_list), np.array(right_x_list), np.array(right_y_list)
