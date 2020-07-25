import requests
import json
import sys
import pandas
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

username = sys.argv[1]
password = sys.argv[2]
AppKey = sys.argv[3]

baseUrl = 'https://ciapi.cityindex.com/TradingApi'

data={"Password": password, "UserName": username, "AppKey":AppKey}

headers = {'Content-type': 'application/json'}
r =requests.post(f'{baseUrl}/session', data=json.dumps(data,indent=2), headers=headers)

authJson = r.json()

authHeaders = {
    "Session": authJson["Session"],
    "UserName" : data["UserName"]
}

marketSearchUrl = f'{baseUrl}/market/search?SearchByMarketName=TRUE&Query=EUR%2FUSD&MaxResults=10'
marketsResult = requests.get(marketSearchUrl, headers=authHeaders)

print(marketsResult.json())
marketId = marketsResult.json()["Markets"][0]["MarketId"]

priceHistoryUrl = f'{baseUrl}/market/{marketId}/barhistory?interval=HOUR&span=4&PriceBars=250'
priceHistory = requests.get(priceHistoryUrl, headers=authHeaders)

priceHistoryDict = priceHistory.json()["PriceBars"]
pricingDataset = pandas.read_json(json.dumps(priceHistoryDict, indent=2))
print(pricingDataset.head(20))
print(pricingDataset.describe())


# box and whisker plots
# pricingDataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# pricingDataset.hist()
# pandas.plotting.scatter_matrix(pricingDataset)
# pyplot.show()

array = pricingDataset.values
X = array[:,1:4]
y = array[:,0]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print(y)

# K fold test is missing something to put values in buckets
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
# 	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# 	results.append(cv_results)
# 	names.append(name)
# 	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
