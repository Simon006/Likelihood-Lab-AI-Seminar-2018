import random as rd
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, input_dim, tree_num, maximal_depth=1000, minimal_samples=1, criterion='gini'):
        # basic classifier information
        self._input_dim = input_dim
        self._tree_num = tree_num
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples
        self._criterion = criterion

        #

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass
