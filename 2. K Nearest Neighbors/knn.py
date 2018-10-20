from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


class Knn:
    def __init__(self, k, x_train, y_train):
        self._k = k
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_test, y_test):
        # Check length error
        if len(x_test) != len(y_test):
            raise ValueError("length doesn't match")

        y_predict = []
        # Loop through the test set
        for sample in x_test:
            # Find K nearest neighbors
            neighbor_list = self._neighbor_search(sample)

            # Majority voting
            predict_result = self._vote(neighbor_list)
            y_predict.append(predict_result)

        # Evaluate
        correct_count = 0
        for i in range(len(y_test)):
            if y_predict[i] == y_test[i]:
                correct_count += 1
            else:
                continue
        accuracy = correct_count / len(y_test)

        return y_predict, accuracy

    def _neighbor_search(self, sample):
        # Calculate sample similarity
        distance_list = []
        for sample_train in self._x_train:
            dist = np.linalg.norm(sample_train - sample)
            distance_list.append(dist)

        # Find neighbors
        distance_rank = np.argsort(distance_list)
        k_nearest_neighbors = distance_rank[:self._k]

        return k_nearest_neighbors

    def _vote(self, neighbor):
        # Find candidate target
        target_list = []
        for index in neighbor:
            target_list.append(self._y_train[index])

        # Voting
        result = max(target_list, key=target_list.count)

        return result


if __name__ == '__main__':

    # Import digits data
    digits = datasets.load_digits()

    # Hyper-parameter
    train_rate = 0.5
    neighbor_num = 2

    # Get input and output
    image_set = digits['data']  # Contains 1797 digit images
    target_set = digits['target']  # Contains the corresponding answers to the digits

    # Split train set and test set
    sample_num = len(image_set)
    x_train_set = image_set[:int(train_rate * sample_num)]
    y_train_set = target_set[:int(train_rate * sample_num)]
    x_test_set = image_set[int(train_rate * sample_num):]
    y_test_set = target_set[int(train_rate * sample_num):]

    # Construct KNN classifier
    knn_classifier = Knn(neighbor_num, x_train_set, y_train_set)

    # Predict and Performance Evaluation
    _, accuracy = knn_classifier.predict(x_test_set, y_test_set)
    print('Prediction Accuracy: ' + str(accuracy))
