import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score, jaccard_similarity_score


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

    cols = cell_data.columns
    print(cols)
    x = np.asarray(cell_data[cols[1:-1]])
    y = np.asarray(cell_data[cols[-1]])

    print(x.shape)
    print(y.shape)
    return train_test_split(x, y, train_size=0.8, random_state=3)


def train_model(x_train, y_train, kernel='rbf'):
    model = svm.SVC(kernel=kernel)
    model.fit(x_train, y_train)

    return model


def predict(model, x_test):
    return model.predict(x_test)


def evaluate(y_hat, y):
    print(f"f1-score: {f1_score(y, y_hat, average='weighted')}")
    print(f"Jaccard score: {jaccard_similarity_score(y, y_hat)}")


if __name__ == '__main__':
    data = read_data("cell_samples.csv")
    # distro(data)
    x_train, x_test, y_train, y_test = process_data(data)
    print(f"Train set: {x_train.shape}, {y_train.shape}")
    print(f"Test set: {x_test.shape}, {y_test.shape}")

    model_rbf = train_model(x_train, y_train)
    y_hat = predict(model_rbf, x_test)
    evaluate(y_hat, y_test)

    model_linear = train_model(x_train, y_train, 'linear')
    y_hat_linear = predict(model_linear, x_test)
    evaluate(y_hat_linear, y_test)
