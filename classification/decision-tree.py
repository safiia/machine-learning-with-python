import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier


def read_data(filename):
    data = pandas.read_csv(filename, delimiter=',')
    return data


if __name__ == '__main__':
    data = read_data("drug200.csv")