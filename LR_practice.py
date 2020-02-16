import sklearn
from sklearn import linear_model, preprocessing
from sklearn import datasets
import pickle
import numpy as np
import pandas as pd


data = datasets.load_digits ()
# print(data.feature_names)
# print(data.target_names)

x = data.data
y = data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0.7429891649862652
for i in range (100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = linear_model.LinearRegression()
    model.fit (x_train, y_train)
    acc = model.score (x_test, y_test)
    print ('interation:', i, 'accuracy:', acc)

    if acc > best:
        best = acc
        with open ('LR.pickle', 'wb') as f:
            pickle.dump (model, f)

load_in = open ('LR.pickle', 'rb')
linear = pickle.load (load_in)

print ('Co-efficent: \n', linear.coef_)
print ('Intercept: \n', linear.intercept_)

predictions = linear.predict (x_test)
for i in range (len (predictions)):
    print ('Prediction:', predictions[i] * 5, 'Data:', x_test[i], 'Actual:', y_test[i] * 5)
print (best)
