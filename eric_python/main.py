import math

import numpy as np
import pandas
import pandas as pd
import requests
import matplotlib.pyplot as plt
from math import floor

from keras import Sequential, optimizers, regularizers
from keras.layers import MaxPool2D, Dropout, Conv2D, Flatten, Dense
from sklearn.metrics import accuracy_score
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
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
from multiprocessing import Process
# Requests

# Constants
HOLD = 0
BUY = 1
SELL = 2

def configure_full_df_print():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        # print(type(x), type(x_temp), x.shape)
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp

def label_df(dframe, prices, windowSize, numDays):
    winBegin = 0
    winEnd = winBegin + windowSize
    countRow = 0
    labels = np.zeros([numDays])
    while countRow <= numDays:
        minValue = min(prices[winBegin:winEnd])
        maxValue = max(prices[winBegin:winEnd])
        for i, val in enumerate(prices[winBegin:winEnd]):
            finalInd = i + winBegin
            if val == minValue and val is not None:
                labels[finalInd] = HOLD
                labels[finalInd+1] = BUY
            elif val == maxValue and val is not None:
                labels[finalInd] = HOLD
                labels[finalInd+1] = SELL
            elif val is not None:
                labels[finalInd] = HOLD
        winBegin = winEnd + 1
        winEnd = winBegin + windowSize
        countRow = winEnd
    dframe['y'] = labels
    dframe['y_rounded'] = round(dframe['y']).astype(int)
    dframe['y'] = dframe['y_rounded']
    dframe.drop('y_rounded', axis=1, inplace=True)
    return dframe

def build_model(full_data):
    params = {'batch_size': 80,
              'conv2d_layers': {'conv2d_do_1': 0.2, 'conv2d_filters_1': 32, 'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0,
                                'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.3,
                                'conv2d_filters_2': 64, 'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2,
                                'conv2d_strides_2': 1,
                                'kernel_regularizer_2': 0.0, 'layers': 'two'},
              'dense_layers': {'dense_do_1': 0.3, 'dense_nodes_1': 128, 'kernel_regularizer_1': 0.0, 'layers': 'one'},
              'epochs': 30000, 'lr': 0.05, 'optimizer': 'adam'}
    full_data_no_y = full_data.drop(columns="y")

    tscv = TimeSeriesSplit(n_splits=9)
    for tr_index, test_index in tscv.split(full_data_no_y.values):
        X_train_full, X_test = full_data_no_y.values[tr_index], full_data_no_y.values[test_index]
        y_train_full, y_test = full_data['y'].values[tr_index], full_data['y'].values[test_index]

    for tr_index, val_index in tscv.split(X_train_full):
        X_train, X_valid = X_train_full[tr_index], X_train_full[val_index]
        y_train, y_valid = y_train_full[tr_index], y_train_full[val_index]


    print("model X_train shape", X_train.shape)
    X_train = reshape_as_image(X_train, 3, 3)
    X_valid = reshape_as_image(X_valid, 3, 3)
    X_test = reshape_as_image(X_test, 3, 3)
    X_train = np.stack((X_train,) * 3, axis=-1)
    X_test = np.stack((X_test,) * 3, axis=-1)
    X_valid = np.stack((X_valid,) * 3, axis=-1)
    full_data_no_y = full_data_no_y.to_numpy()
    full_data_no_y = reshape_as_image(full_data_no_y, 3, 3)
    full_data_no_y = np.stack((full_data_no_y,) * 3, axis=-1)
    model = Sequential()

    print("Training with params {}".format(params))

    conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                           params["conv2d_layers"]["conv2d_kernel_size_1"],
                           strides=params["conv2d_layers"]["conv2d_strides_1"],
                           kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                           padding='same', activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(X_train[0].shape[0],
                                        X_train[0].shape[1], X_train[0].shape[2]))
    model.add(conv2d_layer1)
    if params["conv2d_layers"]['conv2d_mp_1'] > 1:
        model.add(MaxPool2D(pool_size=params["conv2d_layers"]['conv2d_mp_1']))

    model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
    if params["conv2d_layers"]['layers'] == 'two':
        conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                               params["conv2d_layers"]["conv2d_kernel_size_2"],
                               strides=params["conv2d_layers"]["conv2d_strides_2"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                               padding='same', activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform')
        model.add(conv2d_layer2)

        if params["conv2d_layers"]['conv2d_mp_2'] > 1:
            model.add(MaxPool2D(pool_size=params["conv2d_layers"]['conv2d_mp_2']))

        model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

    model.add(Flatten())

    model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
    model.add(Dropout(params['dense_layers']['dense_do_1']))

    if params['dense_layers']["layers"] == 'two':
        model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                        kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
        model.add(Dropout(params['dense_layers']['dense_do_2']))

    model.add(Dense(3, activation='softmax'))

    if params["optimizer"] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=params["lr"])
    elif params["optimizer"] == 'sgd':
        optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    elif params["optimizer"] == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])

    return model, X_train, X_valid, X_test, X_train, y_train, y_valid, full_data_no_y, y_test
base_url = 'https://cloud.iexapis.com/'
version = 'stable/'

# Add your publishable API token here.
token = config.ERIC_IEX_TOKEN

# Specify that youʼre retrieving a specific value from the Key Stats endpoint.
symbol_param = 'CVX' # Chevron
field_param = 'marketcap'
#endpoint_path = f'stock/{symbol_param}/indicator/di?range=5dm' # https://sandbox.iexapis.com/stable/stock/JNJ/stats?token={token}
# Endpoint_path = f'stock/{symbol_param}/stats/{field_param}'
query_params = f'?token={token}'
endpoint_path = f'stock/{symbol_param}/chart/date/20220203'
api_call = f'{base_url}{version}{endpoint_path}{query_params}'
print(f'API Call: {api_call}')

r = requests.get(api_call) # Make HTTPS call
#print("REQUESTS STATUS CODE IS ", r.status_code)
#data = r.json() # Decode JSON

#a_file = open("data_regular_by_minute.json", "w")
#json.dump(data, a_file)
#print(f'Headers: {r.headers}') # Show headers
#print(f"IEX Cloud Credits Used: {r.headers['iexcloud-credits-used']}")
#print(f'Data: {data}') # Print decodedJSON object
data = open("data_regular_max.json", "r")
a_dictionary = json.load(data)
#print(json.dumps(a_dictionary, sort_keys=True, indent=4))


data_pd = pd.json_normalize(a_dictionary)
print(data_pd)
#data_pd['date'] = data_pd['date'] + ' ' + data_pd['minute']
data_pd['date'] = pd.to_datetime(data_pd['date'])
#data_pd.drop('minute', axis=1, inplace=True)
data_pd.set_index('date', inplace=True)
print(data_pd.index)
print(data_pd)
df = data_pd

df['Open'] = df['open']
df['High'] = df['high']
df['Low'] = df['low']
df['Close'] = df['close']
df['Volume'] = df['volume']

df = df.dropna()

"""
Assemble a dataframe of technical indicator series for a single stock
"""

df["adx"] = ta.ADX(df['high'],df['low'],df['close'])
df["rsi"] = ta.RSI(df['close'])
df["willr"] = ta.WILLR(df['high'],df['low'],df['close'])
df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'])
configure_full_df_print()
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'adx', 'rsi', 'willr', 'mfi']].copy()
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, columns=X.columns)
X = label_df(X, X['Close'].values, 11, X.shape[0])
df['y'] = X['y'].values
print(X.columns)
print(X)
model, X_train, X_valid, X_test, X_train, y_train, y_valid, X_no_y, y_test = build_model(X)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)
y_pred_full = model.predict(X_no_y)
#print("predshapetrain", y_pred_train.shape)
#print("predshapetest", y_pred_test.shape)
#print("predshapevalid", y_pred_valid.shape)
#print("ypredfullshape", y_pred_full.shape)
#print("closeshape", X['Close'].shape)
#print("Xshape", X.shape)
#print(y_pred_full)
df['predicted_y'] = np.argmax(y_pred_full, axis=1)
#print(df[df['predicted_y'] == 2].shape)
#print(df[df['predicted_y'] == 1].shape)
#print(df[df['predicted_y'] == 0].shape)

#print(df[df['y'] == 2].shape)
#print(df[df['y'] == 1].shape)
#print(df[df['y'] == 0].shape)
#print(df['y'])

print(df[['y','predicted_y']])
print("NUM Ys IS", len(df['y']))
print("ACCURACY SCORE", accuracy_score(df['y'], df['predicted_y']))

use_ml = True
class ParabolicStrategy(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.close
        #print("---------------CLOSE---------")
        #print(self.close)
        self.adx = self.I(ta.ADX, self.high, self.low, self.close)
        self.sar = self.I(ta.SAR, self.high, self.low)
        #print("----------SAR-------------")
        #print(self.sar)
        self.pdi = self.I(ta.PLUS_DI, self.high, self.low, self.close)
        self.mdi = self.I(ta.MINUS_DI, self.high, self.low, self.close)
        self.rsi = self.I(ta.RSI, self.close)
        self.pred_y = self.data.predicted_y

    def next(self):
        #print("SELF ADX -2 -1 0 [", self.adx[-2], self.adx[-1], self.adx[0], "]")
        #print("NEXT PREDICTED ACTION GENERAL", self.pred_y[0])
        if ((self.adx[-1] > 25 and self.adx[-2] <= 25 and self.pdi[-1] > self.mdi[-1]) or (use_ml and self.pred_y[0] == BUY)) and not self.position.is_long:
            #if use_ml and self.pred_y[0] == BUY:
                #print("ML BUY!!!!!")
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.buy(size=0.2, tp=350, limit=100)
        if ((self.adx[-1] > 25 and self.adx[-2] <= 25 and self.mdi[-1] > self.pdi[-1] and self.rsi[-1] < 30) or (use_ml and self.pred_y[0] == SELL)) and not self.position.is_short:
            #if use_ml and self.pred_y[0] == SELL:
                #print("ML SELL!!!!!")
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.sell(size=0.2, tp=200, limit=300)
        # close buy position when price moves below PSAR and close sell position when price moves below psar
        if self.position.is_long and self.sar[-1] > self.close[-1] and self.sar[-2] <= self.close[-2]:
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.position.close()
        if self.position.is_short and self.sar[-1] < self.close[-1] and self.sar[-2] >= self.close[-2]:
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.position.close()



bt = Backtest(df, ParabolicStrategy, cash=100_000, commission=.002)
stats = bt.run()
print(stats)

def expectunity(stats):
    numTrades = stats['# Trades']
    tradeDays = stats['Avg. Trade Duration'].days * numTrades
    expectancy = stats['Expectancy [%]'] / 100.0
    opportunities = numTrades * 365 / tradeDays
    return expectancy * opportunities


print(expectunity(stats))

class DummyParabolicStrategy(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.close
        #print("---------------CLOSE---------")
        #print(self.close)
        self.adx = self.I(ta.ADX, self.high, self.low, self.close)
        self.sar = self.I(ta.SAR, self.high, self.low)
        #print("----------SAR-------------")
        #print(self.sar)
        self.pdi = self.I(ta.PLUS_DI, self.high, self.low, self.close)
        self.mdi = self.I(ta.MINUS_DI, self.high, self.low, self.close)
        self.rsi = self.I(ta.RSI, self.close)
        self.pred_y = self.data.predicted_y

    def next(self):
        #print("SELF ADX -2 -1 0 [", self.adx[-2], self.adx[-1], self.adx[0], "]")
        #print("NEXT PREDICTED ACTION GENERAL", self.pred_y[0])
        if ((self.adx[-1] > 25 and self.adx[-2] <= 25 and self.pdi[-1] > self.mdi[-1]) or (use_ml and self.pred_y[0] == BUY)) and not self.position.is_long:
            #if use_ml and self.pred_y[0] == BUY:
                #print("ML BUY!!!!!")
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.buy(size=0.2, tp=350, limit=100)
        if ((self.adx[-1] > 25 and self.adx[-2] <= 25 and self.mdi[-1] > self.pdi[-1] and self.rsi[-1] < 30) or (use_ml and self.pred_y[0] == SELL)) and not self.position.is_short:
            #if use_ml and self.pred_y[0] == SELL:
                #print("ML SELL!!!!!")
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.sell(size=0.2, tp=200, limit=300)
        # close buy position when price moves below PSAR and close sell position when price moves below psar
        if self.position.is_long and self.sar[-1] > self.close[-1] and self.sar[-2] <= self.close[-2]:
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.position.close()
        if self.position.is_short and self.sar[-1] < self.close[-1] and self.sar[-2] >= self.close[-2]:
            #print("NEXT PREDICTED ACTION", self.pred_y[0])
            self.position.close()

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

class IntegrationClass:
    cash = 0
    commission = 0
    # default constructor
    def __init__(self, cash, commission):
        self.cash = cash
        self.commission = commission
    def backtestObj(self, df, strat, cash, commission):
        return Backtest(df, strat, cash=cash, commission=commission)
    def statsForBacktestObj(self, obj):
        return obj.run()
    def run(self):
        btOne = self.backtestObj(df, DummyParabolicStrategy, self.cash/2, self.commission)
        btTwo = self.backtestObj(df, ParabolicStrategy, self.cash/2, self.commission)

        pool = multiprocessing.Pool(processes=3)
        args = [btOne, btTwo]
        statsresults = pool.map(self.statsForBacktestObj, args)
        print("STATS RESULTS IS ", statsresults)
        print("STATS LEN IS", len(statsresults))
        #statsOne = self.statsForBacktestObj(btOne)
        #statsTwo = self.statsForBacktestObj(btTwo)


#f = open("DS598 Code in Python.R", "r")
#print(f.read())

if __name__ == '__main__':
    integration_obj = IntegrationClass(cash=1000000, commission=0.02)
    integration_obj.run()