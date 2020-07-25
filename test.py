import requests
import json
import sys

username = sys.argv[1]
password = sys.argv[2]
AppKey = sys.argv[3]

data={"Password": password, "UserName": username, "AppKey":AppKey}


headers = {'Content-type': 'application/json'}
r =requests.post('https://ciapi.cityindex.com/TradingApi/session', data=json.dumps(data,indent=2), headers=headers)

authJson = r.json()
print(authJson)

authHeaders = {
    "Session": authJson["Session"],
    "UserName" : data["UserName"]
}

marketSearchUrl = 'https://ciapi.cityindex.com/TradingApi/market/search?SearchByMarketName=TRUE&Query=EUR%2FUSD&MaxResults=10'
marketsResult = requests.get(marketSearchUrl, headers=authHeaders)

print(marketsResult.text)