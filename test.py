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

def convertDate(datestr):
    datetimestr = re.sub(r'/Date\(', '', datestr)
    datetimestr = re.sub(r'\)/', '',datetimestr)
    dt = datetime.fromtimestamp(int(datetimestr)/1000.0)
    return dt

def queryDataSet(username, password, AppKey):
    baseUrl = 'https://ciapi.cityindex.com/TradingApi'
    data={"Password": password, "UserName": username, "AppKey":AppKey}
    headers = {'Content-type': 'application/json'}
    r=requests.post(f'{baseUrl}/session', data=json.dumps(data,indent=2), headers=headers)

    authJson = r.json()

    authHeaders = {
        "Session": authJson["Session"],
        "UserName" : data["UserName"]
    }

    marketSearchUrl = f'{baseUrl}/market/search?SearchByMarketName=TRUE&Query=EUR%2FUSD&MaxResults=10'
    marketsResult = requests.get(marketSearchUrl, headers=authHeaders)

    #   print(marketsResult.json())
    marketId = marketsResult.json()["Markets"][0]["MarketId"]

    priceHistoryUrl = f'{baseUrl}/market/{marketId}/barhistory?interval=HOUR&span=4&PriceBars=250'
    priceHistory = requests.get(priceHistoryUrl, headers=authHeaders)

    priceHistoryDict = priceHistory.json()["PriceBars"]
    pricingDataset = pandas.read_json(json.dumps(priceHistoryDict, indent=2))
    pricingDataset["positive_direction"] = numpy.where(pricingDataset['Close'] > pricingDataset['Open'], int(1), int(0))
    pricingDataset["BarDate"] = pricingDataset["BarDate"].apply(lambda x: convertDate(x))
    return pricingDataset


username = sys.argv[1]
password = sys.argv[2]
AppKey = sys.argv[3]
pricingDataset = queryDataSet(username, password, AppKey)

# print(pricingDataset.head(10))
# print(pricingDataset.describe())
closeAndTimeDataSet = pricingDataset.drop(['Open', 'High', 'Low', 'positive_direction'], axis=1)
closeAndTimeDataSet.index = closeAndTimeDataSet.BarDate
closeAndTimeDataSet = closeAndTimeDataSet.drop('BarDate', axis=1)
# print(closeAndTimeDataSet.head(5))

# closeAndTimeDataSet.plot()
# pyplot.show()

# # box and whisker plots
# # pricingDataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # pricingDataset.hist()
# # pandas.plotting.scatter_matrix(pricingDataset)
# # pyplot.show()

print(closeAndTimeDataSet.describe())
array = closeAndTimeDataSet.values

training_data_len = math.ceil(len(array) * .8 )
print(training_data_len)

# Scaling the data to normalize the input data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(array)

print(scaled_data)

# Create training dataset
# create scaled dataset
train = array[0:training_data_len,:]
#split data to x_train and y_train datasets
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = numpy.array(x_train), numpy.array(y_train)
print(x_train.shape)
x_train = numpy.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

#Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#Create testing dataset
#Create new array from lentgth-60 to length of scaled data
test_data = scaled_data[training_data_len - 60:,:]

#Create datasets x_test and y_test
x_test = []
y_test = array[training_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Convert data to numpy array
x_test = numpy.array(x_test)

#Reshape the data
x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get model predicted price values
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

# Use root mean squared error (RMSE)
rmse = numpy.sqrt(numpy.mean(prediction - y_test)**2)
print(rmse)

#Plot the data
train = closeAndTimeDataSet[:training_data_len]
valid = closeAndTimeDataSet[training_data_len:]
valid['Predictions'] = prediction

#Visualize the data
pyplot.figure(figsize=(16,8))
pyplot.title("model")
pyplot.xlabel('Date', fontsize=18)
pyplot.ylabel('Close price', fontsize=18)
pyplot.plot(train['Close'])
pyplot.plot(valid[['Close','Predictions']])
pyplot.legend(['Train','Val', 'Predictions'], loc='lower right')
pyplot.show()

print(valid)

# Predicting the price for the next day
last_60_days = closeAndTimeDataSet[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = numpy.array(X_test)

X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)

print(pred_price)

# X = array[:,1:5]
# y = array[:,5]
# train = array[0:100,:]
# valid = array[100:,:]
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(closeAndTimeDataSet)

# x_train, y_train = [], []
# for i in range(60,len(train)):
#     # print(scaled_data[i-1:i,0])
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
# x_train, y_train = numpy.array(x_train), numpy.array(y_train)

# x_train = numpy.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
# # print(x_train)
# # print(y_train)

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))

# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# #predicting 246 values, using past 60 from the train data
# inputs = closeAndTimeDataSet[len(closeAndTimeDataSet) - len(valid) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs  = scaler.transform(inputs)

# X_test = []
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
# X_test = numpy.array(X_test)


# X_test = numpy.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# closing_price = model.predict(X_test)
# closing_price = scaler.inverse_transform(closing_price)
# print(len(closing_price))
# # print(valid)

# # rms=numpy.sqrt(numpy.mean(numpy.power((valid-closing_price),2)))
# # print(rms)
# # 11.772259608962642
# #for plotting
# train = closeAndTimeDataSet[:100]
# valid = closeAndTimeDataSet[100:]
# valid['Predictions'] = closing_price
# pyplot.plot(train['Close'])
# pyplot.plot(valid[['Close','Predictions']])
# pyplot.show()

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
# models.append(('RFR', RandomForestRegressor(n_estimators = 1000, random_state = 1)))

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

# # 
# model = RandomForestRegressor(n_estimators = 1000, random_state = 1)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)

# # print(X_validation)
# # print(predictions)

# preds = numpy.asarray([1 if val > 0.5 else 0 for val in predictions])

# print(preds)
# print(Y_validation)
# print(accuracy_score(Y_validation, preds))
# # print(confusion_matrix(Y_validation, preds))
# # print(classification_report(Y_validation, preds))

# # # Evaluate predictions
# # print(accuracy_score(Y_validation, predictions))
# # print(confusion_matrix(Y_validation, predictions))
# # print(classification_report(Y_validation, predictions))