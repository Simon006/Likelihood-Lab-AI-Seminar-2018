import numpy as np
import random as rd
from math import log
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

        # define the list of weak trees
        self._weak_trees = []

    def train(self, x, y):
        # initialize the weights uniformly
        sample_weights = np.ones(len(x)) / len(x)

        while len(self._weak_trees) < self._tree_num:
            # define weak learner
            weak_learner = dict()
            weak_learner['weight'] = None
            weak_learner['model'] = DecisionTree(input_dim=self._input_dim, maximal_depth=self._maximal_depth,
                                                 minimal_samples=self._minimal_samples, criterion=self._criterion)

            # sample(with replacement) training data-set with respect to sample_weights
            x_sampled, y_sampled = ...

            # train the weak learner
            weak_learner.train(x_sampled, y_sampled)

            # calculate the weak learner's weight
            error = 1 - weak_learner.evaluate(x_sampled, y_sampled)
            weak_learner['weight'] = 0.5 * log((1 - error) / error)


    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass
