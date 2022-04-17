
import math
import os
import numpy as np
import pandas
import pandas as pd
import requests
import csv
import matplotlib
import matplotlib.pyplot as plt
from math import floor
import seaborn as sns
import datetime
from datetime import date, timedelta
from keras import Sequential, optimizers, regularizers
from keras.layers import MaxPool2D, Dropout, Conv2D, Flatten, Dense
from termcolor import colored as cl
from datetime import timedelta
from datetime import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy
import config
import json
import talib as ta
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import backend as K


#os.chdir("/Users/ridhimasaxena/Desktop/GQP/PythonCode")

#####
base_url = 'https://cloud.iexapis.com/'
version = 'stable/'

# Add your publishable API token here.
token = config.RIDHIMA_IEX_TOKEN

# Specify that youʼre retrieving a specific value from the Key Stats endpoint.
symbol_param = 'AAPL' # Apple
field_param = 'marketcap'
endpoint_path = f'stock/{symbol_param}/indicator/di?range=1mm' # https://sandbox.iexapis.com/stable/stock/JNJ/stats?token={token}
Endpoint_path = f'stock/{symbol_param}/stats/{field_param}'
query_params = f'?token={token}'
endpoint_path = f'stock/{symbol_param}/chart/date/20220131'
api_call = f'{base_url}{version}{endpoint_path}{query_params}'
print(f'API Call: {api_call}')

### To Get 30 stock days ###

#n is number of days you want
# def getPreviousDays(n):
#     dates = []
#     ctr = 0
#     for i in range(0, n+n):
#         days_before = (date.today() - timedelta(days=i)).isoformat()
#         # This just means if not Saturday or sunday put it in
#         if (date.today() - timedelta(days=i)).weekday() < 5:
#             #print("WEEKDAY IS", (date.today() - timedelta(days=i)).weekday())
#             #dates.append(days_before)
#             dates.append(days_before.replace('-', ''))
#             ctr = ctr+1
#             #print("CTR IS", ctr, " I IS", i)
#         if ctr == n:
#             print(dates)
#             return dates
#
#
# Dates = getPreviousDays(30)
# data = []
# for i in Dates:
#     endpoint_path = f'stock/{symbol_param}/chart/date/' + i
#     api_call = f'{base_url}{version}{endpoint_path}{query_params}'
#     r = requests.get(api_call)
#     data.append(r.json())
#     print(endpoint_path)

#r = requests.get(api_call) # Make HTTPS call
#print("REQUESTS STATUS CODE IS ", r.status_code)
# data = r.json() # Decode JSON

#print(type(data))
# To create json file and write data to json file

#a_file = open("data_regular_by_minute_aapl.json", "a") # whatever file I want to append to
#json.dump(data, a_file)
#print(f'Headers: {r.headers}') # Show headers
#print(f"IEX Cloud Credits Used: {r.headers['iexcloud-credits-used']}")
#print(f'Data: {data}') # Print decodedJSON object
data = open("data_regular_by_minute_aapl.json", "r") #the file that I want to read
a_dictionary = json.load(data)
#print(json.dumps(a_dictionary, sort_keys=True, indent=4))

# Creating the data frame
data_pd = pd.json_normalize(a_dictionary)
print(data_pd)
data_pd['date'] = data_pd['date'] + ' ' + data_pd['minute']
data_pd['date'] = pd.to_datetime(data_pd['date'])
data_pd.drop('minute', axis=1, inplace=True)
data_pd.set_index('date', inplace=True)
# print(data_pd.index)
# print(data_pd)
df = data_pd


df['Open'] = df['open']
df['High'] = df['high']
df['Low'] = df['low']
df['Close'] = df['close']
df['Volume'] = df['volume']
# print(type(ta.STOCHF(df['high'], df['low'], df['close'])))
df = df.dropna()
df["stochastic_k"], df["stochastic_d"] = ta.STOCHF(df['high'], df['low'], df['close'])
df["macd"], df["macd_signal"], df["macd_hist"] = ta.MACD(df['close'].values)
df["RSI"] = ta.RSI(df['close'])
df["RSI_Average"] = ta.RSI(df['average'])
df["ClosePrior"] = df[['Close']].shift(-1)
df['Classification'] = 0


#if df["macd_signal"] > df["macd"] > 0 and df["stochastic_k"] < 20 and 40 > df["RSI"] > 30 and df["ClosePrior"] > df['Close']:
#    df["Classification"] = 1
# and row["stochastic_k"] < 20 and 40 > row["RSI"] > 30 and row["ClosePrior"] > row['Close']


for i, row in df.iterrows():
    ifor_val = 0
    if row["macd"] > row["macd_signal"] and row["macd"]> 0 and row["stochastic_k"] < 30 or 52 > row["RSI"] > 25 and row["ClosePrior"] > row['Close']:
        ifor_val = 1
    df.at[i,'Classification'] = ifor_val

#print(df.columns)


df = df[35:7414]
df.to_csv('df_entry.csv')

## Entry Strategy
#Create SVM Model

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'stochastic_k', 'macd','macd_signal', 'RSI', 'ClosePrior']].copy()
Y = df[['Classification']].copy()
x = X.values #returns a numpy array
y = Y.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
y_scaled = min_max_scaler.fit_transform(y)
X = pd.DataFrame(x_scaled, columns=X.columns)
Y = pd.DataFrame(y_scaled, columns=Y.columns)

print(X.columns)
print(X)
print(Y.columns)
print(Y)

X = X.dropna()
Y = Y.dropna()
#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=False, random_state= 101)

from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state = 1)
SVM = model.fit(X_train, Y_train)
predictions = model.predict(X_test).reshape(-1,1)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Y_test, predictions))


#Hyperparameter Tuning using Gridsearch
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, Y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(Y_test, grid_predictions))

## Exit Strategy

#yoyo strategy

df['ATR'] = ta.ATR(df['high'],df['low'],df['close'])

df['yoyo'] = df['Close'] - 2 * df['ATR']


#adx
df['adx'] = ta.ADX(df['high'],df['low'],df['close'])
df["adx_prior"] = df[['adx']].shift(-1)
df["adx_2prior"] = df[['adx']].shift(-2)

df['Exit'] = 0

for i, row in df.iterrows():
    ifor_val = 0
    if row["adx_prior"] > 25 and row["adx_2prior"] <= 25 or row['Close'] <= row['yoyo']:
        ifor_val = 1
    df.at[i, 'Exit'] = ifor_val

df.to_csv('df_exit.csv')

# Backtesting

class Testing(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.close
        self.adx = self.I(ta.ADX, self.high, self.low, self.close)


    def next(self):
        if df['Classification'][-1] == 1:
            self.buy(size=0.2, tp=350, limit=100)

        if df['Exit'][-1] == 1:
            self.position.close()


bt = Backtest(df, Testing, cash=100_000, commission=.002)
#stats = bt.run()



def expectunity(numTrades, tradeDats, expectancy):
    numTrades = 114
    tradeDays = 32
    expectancy = 1.31
    opportunities = numTrades * 365 / tradeDays #1300.31
    return expectancy * opportunities #1705.35


print(expectunity)



