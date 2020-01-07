import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


def read_data(filename):
    df = pandas.read_csv(filename)
    print(df.head(9))
    x_data = df.Year
    y_data = df.Value
    return x_data, y_data


def plot_data(x, y, title):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'ro')
    plt.ylabel("GDP")
    plt.xlabel("Year")
    plt.title(title)
    plt.show()


def plot_fit_curve(x, y, params):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, y, '-ro', label='data')
    opt_curve = sigmoid(x, *params)
    plt.plot(x, opt_curve, '-g', label='fit')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.title("Data with Fitted Curve")
    plt.show()


def logistic_function():
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(-5, 5, 0.1)
    y1 = 1.0 / (1.0 + np.exp(-x))
    y2 = 1.0 / (1.0 + np.exp(x))

    plt.plot(x, y1, '-r', label="Negative beta1")
    plt.plot(x, y2, '-g', label="Positive beta1")
    plt.legend()
    plt.title("Sigmoid Function")
    plt.show()


def sigmoid(x, beta1, beta2):
    y = 1.0 / (1.0 + np.exp(-beta1 * (x - beta2)))
    return y


def fit_curve(x, y):
    popt, pcov = curve_fit(sigmoid, x, y)
    print(f"beta1 = {popt[0]}, beta2 = {popt[1]}")
    return popt


def accuracy(x, y):
    # divide data to train/test, learn on train, predict on test
    mask = np.random.rand(len(x)) < 0.8
    x_train = x[mask]
    y_train = y[mask]

    x_test = x[~mask]
    y_test = y[~mask]

    opt_params = fit_curve(x_train, y_train)
    plot_fit_curve(x_train, y_train, opt_params)

    y_hat = sigmoid(x_test, *opt_params)
    print(f"MAE = {np.mean(np.abs(y_test-y_hat))}")
    print(f"MSE = {np.mean((y_test - y_hat)**2)}")
    # print(f"Accuracy = {accuracy_score(y_test, y_hat)}")
    print(f"R2 Score = {r2_score(y_test, y_hat)}")


if __name__ == '__main__':
    x_data, y_data = read_data("china_gdp.csv")
    plot_data(x_data, y_data, "GDP")
    logistic_function()

    # normalize the data
    x_norm = x_data / np.max(x_data)
    y_norm = y_data / np.max(y_data)
    plot_data(x_norm, y_norm, "Normalized GDP Data")

    # find optimal parameters for sigmoid function
    opt_params = fit_curve(x_norm, y_norm)
    plot_fit_curve(x_norm, y_norm, opt_params)

    # find the accuracy of model
    accuracy(x_norm, y_norm)

