import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_data(file_name):
    data = pandas.read_csv(file_name)
    print(data.head(9))
    # print(data['Class'].value_counts())
    return data


def distro(data):
    ax = data[data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
                                             label='malignant')
    data[data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',
                                              ax=ax)
    plt.show()


def process_data(data):
    print(data.dtypes)
    cell_data = data[pandas.to_numeric(data['BareNuc'], errors='coerce').notnull()]
    cell_data['BareNuc'] = cell_data['BareNuc'].astype('int')
    print(cell_data.dtypes)

    return cell_data


if __name__ == '__main__':
    data = read_data("cell_samples.csv")
    # distro(data)
    data = process_data(data)
