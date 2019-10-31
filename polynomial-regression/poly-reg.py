import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


def read_data(filename):
    df = pandas.read_csv(filename)
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    return cdf


def scatter_plot(x, y, regr, deg=2, col='green'):
    plt.scatter(x, y, color=col)
    plt.xlabel("Engine size")
    plt.ylabel("CO2 Emission")

    xx = np.arange(0, 10, 0.01)
    yy = regr.intercept_[0] + regr.coef_[0][1] * xx + regr.coef_[0][2] * xx ** 2
    if deg == 3:
        yy += regr.coef_[0][3] * xx**3
    plt.plot(xx, yy, 'r')
    plt.show()


def split_train_test(cdf, ratio):
    mask = np.random.rand(len(cdf)) < ratio
    train = cdf[mask]
    test = cdf[~mask]
    return train, test


def train_model(train, deg=2):
    x = np.asanyarray(train[['ENGINESIZE']])
    y = np.asanyarray(train[['CO2EMISSIONS']])

    poly = PolynomialFeatures(deg)
    poly_x = poly.fit_transform(x)

    regr = linear_model.LinearRegression()
    regr.fit(poly_x, y)

    print(f"Intercept: {regr.intercept_}")
    print(f"Coefficients: {regr.coef_}")

    scatter_plot(x, y, regr, deg)

    return regr


def predict(test, regr, deg=2):
    x = np.asanyarray(test[['ENGINESIZE']])
    y = np.asanyarray(test[['CO2EMISSIONS']])

    poly = PolynomialFeatures(deg)
    poly_x = poly.fit_transform(x)
    y_hat = regr.predict(poly_x)

    print(f"Mean Absolute Error: {np.mean(np.abs(y - y_hat))}")
    print(f"Mean Squared Error: {np.mean((y - y_hat)**2)}")
    print(f"R2-score: {metrics.r2_score(y, y_hat)}")


if __name__ == '__main__':
    cdf = read_data("../simple-linear-regression/FuelConsumption.csv")
    train, test = split_train_test(cdf, 0.8)

    degree = [2, 3]
    for deg in degree:
        print(f"Polynomial degree {deg}")
        model = train_model(train, deg)
        predict(test, model, deg)




