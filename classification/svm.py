import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_data(file_name):
    data = pandas.read_csv(file_name)
    return data


if __name__ == '__main__':
    data = read_data("cell_samples.csv")