import pandas as pd
from pandas import DataFrame
import numpy as np
import pandas_datareader
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import sklearn.linear_model
import sklearn.model_selection
import pprint


digits = datasets.load_digits()
print(dir(digits))
print(digits.images.shape)
print(digits.images[0])
# plt.matshow(digits.images[0], cmap = 'Greys')
# plt.show()
print(digits.target[0])

n_train = len(digits.data)*2//3
x_train = digits.data[:n_train]
y_train = digits.target[:n_train]
x_test = digits.data[n_train:]
y_test = digits.target[n_train:]
print([d.shape for d in [x_train, y_train, x_test, y_test]])

clf = svm.SVC(gamma=0.001)
clf.fit(x_train,y_train)
print(clf.score(x_test, y_test))

predicted = clf.predict(x_test)
print((y_test != predicted).sum())
print(y_test != predicted)
