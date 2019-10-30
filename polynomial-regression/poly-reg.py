import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def read_data(filename):
    df = pandas.read_csv(filename)
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    return cdf


def split_train_test(cdf, ratio):
    mask = np.random.rand(len(cdf)) < ratio
    train = cdf[mask]
    test = cdf[~mask]
    return train, test


def model(train):
    x = np.asanyarray(train[['ENGINESIZE']])
    y = np.asanyarray(train[['CO2EMISSIONS']])

    poly = PolynomialFeatures(2)
    poly_x = poly.fit_transform(x)


