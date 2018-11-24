import numpy as np
import random as rd
from math import sqrt
from sklearn.datasets import load_wine
from decision_tree import DecisionTree


class BoostingTree:
    def __init__(self, input_dim, tree_num, maximal_depth, minimal_samples, criterion):
        # basic classifier information
        self._input_dim = input_dim
        self._tree_num = tree_num
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples
        self._criterion = criterion

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
