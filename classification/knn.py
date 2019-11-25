import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

    x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
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


def train(x_train, y_train, k=4):
    """
    Train the knn classifier on the training set using k neighbors. This is done by building a KNeighborsClassifier
    that fits the training data and the target values.
    :param x_train:
    :param y_train:
    :param k: number of neighbors to consider
    :return:
    """
    neighbors = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    return neighbors


def predict(neighbors, x_test):
    """
    Predict classes for the test set.
    :param neighbors: KNeighborsClassifier object trained on training data
    :param x_test: test data
    :return:
        y_hat: predicted classes for the test set
    """
    y_hat = neighbors.predict(x_test)
    return y_hat


def accuracy(y_test, y_hat):
    """
    Calculate the accuracy of the model.
    :param y_test: True test set class labels.
    :param y_hat: Predicted class labels for the test set.
    :return:
        the accuracy score of the model.
    """
    return accuracy_score(y_test, y_hat)


def find_best_k(x_train, x_test, y_train, y_test):

    num_k = 10
    accuracies = np.zeros(num_k-1)
    std_acc = np.zeros(num_k-1)

    for k in range(1, num_k):
        model = train(x_train, y_train, k=k)
        y_hat = predict(model, x_test)
        acc = accuracy(y_test, y_hat)
        accuracies[k-1] = acc
        std_acc[k - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])

    print(f"Highest accuracy is for k = {np.argmax(accuracies)} with acc = {np.max(accuracies)}")
    return accuracies, std_acc


def plot_accuracies(acc, std_acc):
    num_k = 10
    plt.plot(range(1, num_k), acc, 'g')
    plt.fill_between(range(1, num_k), acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = read_data("teleCust1000t.csv")
    norm_data = normalize_data(data)
    x_train, x_test, y_train, y_test = split_data(norm_data[0], norm_data[1], 0.8)

    model4 = train(x_train, y_train)
    y_hat = predict(model4, x_test)
    print(f"Model accuracy for k=4: {accuracy(y_test, y_hat)}")

    model6 = train(x_train, y_train, k=6)
    y_hat6 = predict(model6, x_test)
    print(f"Model accuracy for k=6: {accuracy(y_test, y_hat6)}")

    acc, std_acc = find_best_k(x_train, x_test, y_train, y_test)
    plot_accuracies(acc, std_acc)
