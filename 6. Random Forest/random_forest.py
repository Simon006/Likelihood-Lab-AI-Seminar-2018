import numpy as np
import random as rd
from math import sqrt
import multiprocessing as mp
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, input_dim, tree_num, maximal_depth=1000, minimal_samples=1, criterion='gini', cores_num=-1):
        # basic classifier information
        self._forest_input_dim = input_dim
        self._tree_input_dim = int(sqrt(input_dim))
        self._tree_num = tree_num
        self._maximal_depth = maximal_depth
        self._minimal_samples = minimal_samples
        self._criterion = criterion
        if cores_num == -1:
            self._cores_num = mp.cpu_count()
        else:
            self._cores_num = cores_num

        # build forest
        self._forest = self._construct_forest()

    def train(self, x, y):
        pool = mp.Pool(processes=self._cores_num)
        pool.map(func=, iterable=[] * self._cores_num)
        for key in self._forest:
            self._forest[key]['tree'].train(x, y)

    def predict(self, x):
        pool = mp.Pool(processes=self._cores_num)
        pool.map(func=, iterable=[] * self._cores_num)

    def evaluate(self, x, y):
        pass

    def _construct_forest(self):
        forest = dict()
        for i in range(self._tree_num):
            forest[i] = dict()
            forest[i]['tree'] = DecisionTree(self._tree_input_dim, self._maximal_depth,
                                             self._minimal_samples, self._criterion)
            forest[i]['feature'] = self._feature_bagging()
        return forest

    def _feature_bagging(self):
        feature_list = []
        while len(feature_list) < self._tree_input_dim:
            index = rd.randrange(self._forest_input_dim)
            if index not in feature_list:
                feature_list.append(index)
            else:
                continue

        feature_mask = np.zeros(self._forest_input_dim)
        for index in feature_list:
            feature_mask[index] = 1
        feature_mask = np.array(feature_mask, dtype=np.bool)

        return feature_mask


def _train_tree_():
    pass
