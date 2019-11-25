import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


def read_data(filename):
    data = pandas.read_csv(filename, delimiter=',')
    print(f"Size of the data: {data.shape}")

    cols = data.columns
    print(cols)

    # print("Categorical attributes:\n", data['Sex'].value_counts())
    # print("Categorical attributes:\n", data['BP'].value_counts())
    # print("Categorical attributes:\n", data['Cholesterol'].value_counts())

    data = data[cols].values
    x = data[:, :-1]
    y = data[:, -1]
    return x, y, cols


def preprocess_data(col, values):
    """
    Since sklearn cannot process categorical data, we need to convert them to numerical values with sklearn.preprocessing.
    :param col: input data column
    :return:
        transformed data column
    """

    le_col = preprocessing.LabelEncoder()
    le_col.fit(values)
    return le_col.transform(col)


def split_data(x, y, train_size):
    """
    Split the data and the associated labels into training and test sets with a given proportionality.
    :param x: data
    :param y: labels
    :param train_size: proportionality of the train data w.r.t. the whole data
    :return:
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=3, shuffle=True)

    return x_train, x_test, y_train, y_test


def train(x_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(x_train, y_train)

    return model


def predict(model, x_test):
    y_hat = model.predict(x_test)
    return y_hat


def accuracy(y_hat, y_test):
    acc = metrics.accuracy_score(y_test, y_hat)
    # acc = np.sum(y_hat==y_test) / y_test.shape[0]
    return acc


def visualize_tree(model, y, y_train, cols):
    dot_data = StringIO()
    filename = "drug_tree.png"
    feature_names = cols[0:5]
    target_names = np.unique(y).tolist()
    out = tree.export_graphviz(model, feature_names=feature_names, out_file=dot_data,
                               class_names=np.unique(y_train), filled=True, special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img, interpolation='nearest')


if __name__ == '__main__':
    x, y, cols = read_data("drug200.csv")
    x[:, 1] = preprocess_data(x[:, 1], ['F', 'M'])
    x[:, 2] = preprocess_data(x[:, 2], ['LOW', 'NORMAL', 'HIGH'])
    x[:, 3] = preprocess_data(x[:, 3], ['NORMAL', 'HIGH'])

    x_train, x_test, y_train, y_test = split_data(x, y, 0.7)

    model = train(x_train, y_train)
    y_hat = predict(model, x_test)

    print(f"Model accuracy: {accuracy(y_hat, y_test)}")

    visualize_tree(model, y, y_train, cols)






