"""
Likelihood Lab
XingYu
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Node:
    def __init__(self, input_dim, class_num):
        self._input_dim = input_dim
        self._class_num = class_num
        self.split_index = None
        self.split_value = None
        self.leaf_node = None
        self.right_node = None
        self.is_terminal = None
        self.category = None


class DecisionTree:
    def __init__(self, input_dim, class_num, split_criterion, maximal_depth, minimal_samples):
        # Basic Classifier Information
        self._input_dim = input_dim
        self._class_num = class_num
        self._split_criterion = split_criterion
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples

        # Define the tree root
        self._root = Node(input_dim, class_num)

    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

    def _gini(self):
        pass

    def _cross_entropy(self):
        pass
