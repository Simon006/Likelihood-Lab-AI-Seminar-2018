"""
Likelihood Lab
XingYu
"""
from sklearn.datasets import load_iris

class Logistic:
    def __init__(self):


if __name__ == '__main__':

    # Import iris data
    iris = load_iris()

    # Separate input(x) and output(y)
    data_x = iris['data']  # Four attributes: Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、
                           # Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）

    data_y = iris['target']  # Three classes:Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），
                             # Iris Virginica（维吉尼亚鸢尾）

    # Separate training and testing data set
