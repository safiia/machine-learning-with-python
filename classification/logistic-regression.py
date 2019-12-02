import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_score, log_loss
from scipy import optimize
import itertools


def read_data(file_name):
    data = pandas.read_csv(file_name)
    data['churn'] = data['churn'].astype('int')
    churn_df = np.asarray(data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']])
    x = churn_df[:, :-1]
    y = churn_df[:, -1]

    x = preprocessing.StandardScaler().fit(x).transform(x)
    return x, y


def split_data(x, y, train_size):
    return train_test_split(x, y, train_size=train_size, random_state=4)


def train_model(x_train, y_train):
    model = LogisticRegression(C=0.01, solver='liblinear').fit(x_train, y_train)
    return model


def predict(model, x_test):
    y_hat = model.predict(x_test)
    y_hat_prob = model.predict_proba(x_test)
    return y_hat, y_hat_prob


def jaccard_index(y, y_hat):
    return jaccard_score(y, y_hat)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def conf_matrix(y, y_hat):
    return confusion_matrix(y, y_hat, labels=[1, 0])


def model(x, y):
    x_train, x_test, y_train, y_test = split_data(x, y, 0.8)
    m = train_model(x_train, y_train)
    y_hat, y_hat_prob = predict(m, x_test)
    cm = conf_matrix(y_test, y_hat)
    plot_confusion_matrix(cm, classes=['churn=1', 'churn=0'], normalize=False,  title='Confusion matrix')
    print(classification_report(y_test, y_hat))

    print(f"Jaccard score: {jaccard_index(y_test, y_hat)}")
    print(f"Log loss: {log_loss(y_test, y_hat)}")


if __name__ == '__main__':
    x, y = read_data("ChurnData.csv")
    model(x, y)



