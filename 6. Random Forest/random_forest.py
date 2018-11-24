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
        pool.map(func=_train_forest_mp, iterable=[(self._forest, x, y)] * self._cores_num)

    def predict(self, x):
        # each tree evaluates the data set independently.
        y_vote = np.zeros((self._tree_num, len(x)))
        un_voted_list = [True] * self._tree_num
        pool = mp.Pool(processes=self._cores_num)
        pool.map(func=_predict_forest_mp, iterable=[self._forest, x, un_voted_list, y_vote] * self._cores_num)

        # majority voting
        y_predicted = np.zeros(len(x))
        for i in range(len(y_predicted)):
            voted_result = list(y_vote[:,i])
            y_predicted[i] = max(list(voted_result), key=list(voted_result).count)

        return y_predicted

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

    def _construct_forest(self):
        forest = []
        for i in range(self._tree_num):
            tree = dict()
            tree['model'] = DecisionTree(self._tree_input_dim, self._maximal_depth,
                                         self._minimal_samples, self._criterion)
            tree['feature'] = self._feature_bagging()
            tree['untrained'] = True
            forest.append(tree)
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


def _train_forest_mp(forest, train_x, train_y):
    for i in range(len(forest)):
        if forest[i]['untrained']:
            forest[i]['untrained'] = False
            train_x_sampled, train_y_sampled = _sample_with_replacement(train_x, train_y)
            forest[i]['model'].train(train_x_sampled[:,forest[i]['feature']], train_y_sampled)
        else:
            continue


def _predict_forest_mp(forest, x, un_voted_list, y_vote):
    for i in range(len(forest)):
        if un_voted_list[i]:
            un_voted_list[i] = False
            y_vote[i] = forest[i]['model'].predict(x[:,forest[i]['feature']])
        else:
            continue

def _sample_with_replacement(x, y):
    # check dimensionality
    if len(x) != len(y):
        raise ValueError('error.')

    # bootstrap sample
    sample_index_list = [rd.randrange(len(x)) for i in range(len(x))]
    x_sampled = x[sample_index_list]
    y_sampled = y[sample_index_list]

    return x_sampled, y_sampled
