import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing


def read_data(file_name):
    data = pandas.read_csv(file_name)
    cols = data.columns
    print(f"Number of customers in each category: \n{data['custcat'].value_counts()}")

    # convert pandas dataframe to numpy array to use sklearn
    data = data[cols].values
    return data


def normalize_data(data):
    """
    Separate the data from labels and normalize to have zero mean and unit variance
    :param data: data with labels in the last column
    :return:
        X: normalized data
        y: associated class labels
    """

    x = data[:, :-1]
    y = data[:, -1]

    x = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    return x, y


if __name__ == '__main__':
    data = read_data("teleCust1000t.csv")
    normalize_data(data)
    # print(data.head())