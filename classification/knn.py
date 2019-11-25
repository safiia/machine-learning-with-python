import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    print(x[0:5])

    x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
    print(x[0:5])
    return x, y


def split_data(x, y, train_size):
    """
    Split the data and the associated labels into training and test sets with a given proportionality.
    :param x: data
    :param y: labels
    :param train_size: proportionality of the train data w.r.t. the whole data
    :return:
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=4, shuffle=True)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data = read_data("teleCust1000t.csv")
    norm_data = normalize_data(data)
    x_train, x_test, y_train, y_test = split_data(norm_data[0], norm_data[1], 0.8)
