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


def split_train_test(cdf, ratio):
    mask = np.random.rand(len(cdf)) < ratio
    train = cdf[mask]
    test = cdf[~mask]
    return train, test


def train_model(train, fuel):
    regr = linear_model.LinearRegression()
    vars = ['ENGINESIZE', 'CYLINDERS'] + fuel
    x = np.asanyarray(train[vars])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(x, y)
    print(f"Intercept: {regr.intercept_}")
    print(f"Coefficients: {regr.coef_}")
    return regr


def predict(regr, test, fuel):
    vars = ['ENGINESIZE', 'CYLINDERS'] + fuel
    x = np.asanyarray(test[vars])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    y_hat = regr.predict(x)

    print(f"Mean Squared Error for {fuel}: {np.mean((y-y_hat)**2)}")
    print(f"Explained Variance for {fuel}: {regr.score(x, y)}")


if __name__ == '__main__':
    cdf = read_data("../simple-linear-regression/FuelConsumption.csv")
    scatter_plot(cdf, "ENGINESIZE", "CO2EMISSIONS")
    train, test = split_train_test(cdf, 0.8)

    fuels = [['FUELCONSUMPTION_COMB'], ['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
    for fuel in fuels:
        model = train_model(train, fuel)
        predict(model, test, fuel)
        print("\n")
