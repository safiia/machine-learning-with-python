import numpy as np
import matplotlib.pyplot as plt
import pandas
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
plt.show()