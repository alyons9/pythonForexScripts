import requests
import json
import sys
import pandas
import numpy
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import re
import time
from datetime import datetime
from time import mktime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# # box and whisker plots
# # pricingDataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # pricingDataset.hist()
# # pandas.plotting.scatter_matrix(pricingDataset)
# # pyplot.show()

# X = array[:,1:5]
# y = array[:,5]

# # y = pricingDataset.loc[:,["positive_direction"]].values
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# # print(X)
# # K fold test is missing something to put values in buckets
# # Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # models.append(('RFR', RandomForestRegressor(n_estimators = 1000, random_state = 1)))

# # evaluate each model in turn
# # results = []
# # names = []
# # for name, model in models:
# # 	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# # 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# # 	results.append(cv_results)
# # 	names.append(name)
# # 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# # pyplot.boxplot(results, labels=names)
# # pyplot.title('Algorithm Comparison')
# # pyplot.show()

# # model = LinearDiscriminantAnalysis()
# # model.fit(X_train, Y_train)
# # predictions = model.predict(X_validation)
 
# # model = RandomForestRegressor(n_estimators = 1000, random_state = 1)
# # model.fit(X_train, Y_train)
# # predictions = model.predict(X_validation)

# # print(X_validation)
# # print(predictions)

# # # Evaluate predictions
# # print(accuracy_score(Y_validation, predictions))
# # print(confusion_matrix(Y_validation, predictions))
# # print(classification_report(Y_validation, predictions))