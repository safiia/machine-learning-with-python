import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn import metrics
import pylab

df = pandas.read_csv("FuelConsumption.csv")

# take a look at the data
print("Dataset size: ", df.shape)
print(df.head())

# Data Exploration
# take a look at the summary of the data like mean, std, percentiles
print(df.describe())

# select some features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# plot histograms for these features
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
# plt.show()

# plot each feature against Emission to see relation
fig, ax = plt.subplots(1, 3, sharex=False, figsize=(15, 7))
ax[0].scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
ax[0].set_xlabel("FUELCONSUMPTION_COMB")
ax[0].set_ylabel("Emission")

ax[1].scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='red')
ax[1].set_xlabel("ENGINESIZE")
ax[1].set_ylabel("Emission")

ax[2].scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='green')
ax[2].set_xlabel("CYLINDERS")
ax[2].set_ylabel("Emission")

plt.tight_layout()
# plt.show()

# Creating training 80% and test sets 20%
mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

# sklearn to model the simple linear regression
print("Doing linear regression between ENGINESIZE and CO2EMMISIONS...")
regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# print the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# plot learned line on the scatter plot
ax[1].plot(train_x, regr.intercept_[0] + train_x * regr.coef_[0][0], '-k')
# plt.show()

# Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

mae = np.mean(np.abs(test_y - test_y_hat))
mse = np.mean((test_y - test_y_hat)**2)
r2_score = metrics.r2_score(test_y_hat, test_y)

print("Mean Absolute Error: {:.2f}".format(mae))
print("Mean Square Error: {:.2f}".format(mse))
print("R2-score: {:.2f}".format(r2_score))