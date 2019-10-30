import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model


def read_data(filename):
    df = pandas.read_csv(filename)
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
              'CO2EMISSIONS']]
    return cdf


def scatter_plot(cdf, x, y, col='green'):
    plt.scatter(cdf[[x]], cdf[[y]], color=col)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
