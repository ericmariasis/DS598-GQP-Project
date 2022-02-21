import numpy as np
import pandas
import pandas as pd
import requests
import matplotlib.pyplot as plt
from math import floor

from keras import Sequential, optimizers, regularizers
from keras.layers import MaxPool2D, Dropout, Conv2D, Flatten, Dense
from termcolor import colored as cl
from datetime import timedelta
from datetime import datetime
from finta import TA
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
        print("MIN VALUE ISSSS", minValue)
        print("MAX VALUE ISSSS", maxValue)
        for i, val in enumerate(prices[winBegin:winEnd]):
            print("I VAL ISSSSS", i)
            finalInd = i + winBegin
            if val == minValue and val is not None:
                print("GOT TO DA BUY BABYYYYY")
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
    return dframe

def build_model(full_data):
    params = {'batch_size': 80,
              'conv2d_layers': {'conv2d_do_1': 0.2, 'conv2d_filters_1': 32, 'conv2d_kernel_size_1': 3, 'conv2d_mp_1': 0,
                                'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.3,
                                'conv2d_filters_2': 64, 'conv2d_kernel_size_2': 3, 'conv2d_mp_2': 2,
                                'conv2d_strides_2': 1,
                                'kernel_regularizer_2': 0.0, 'layers': 'two'},
              'dense_layers': {'dense_do_1': 0.3, 'dense_nodes_1': 128, 'kernel_regularizer_1': 0.0, 'layers': 'one'},
              'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam'}
    full_data_no_y = full_data.drop(columns="y")
    X_train_full, X_test, y_train_full, y_test = train_test_split(full_data_no_y.values, full_data['y'].values, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
    X_train = reshape_as_image(X_train, 3, 3)
    X_valid = reshape_as_image(X_valid, 3, 3)
    X_test = reshape_as_image(X_test, 3, 3)
    X_train = np.stack((X_train,) * 3, axis=-1)
    X_test = np.stack((X_test,) * 3, axis=-1)
    X_valid = np.stack((X_valid,) * 3, axis=-1)
    print("XTRAIN SHAPE IS", X_train.shape)
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

    return model, X_train, X_valid, X_test, X_train, y_train, y_valid
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

#r = requests.get(api_call) # Make HTTPS call
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
print("DOSE X COLUMNSSSSSS!!!!!")
print(X.columns)
print(X)
X = label_df(X, X['Close'].values, 11, X.shape[0])
print(X.columns)
print(X)
model, X_train, X_valid, X_test, X_train, y_train, y_valid = build_model(X)
print("XTEST SHAPE", X_valid.shape)
y_pred = model.predict(X_test)
print(y_pred)
exit()

class ParabolicStrategy(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.close
        print("---------------CLOSE---------")
        print(self.close)
        self.adx = self.I(ta.ADX, self.high, self.low, self.close)
        self.sar = self.I(ta.SAR, self.high, self.low)
        print("----------SAR-------------")
        print(self.sar)
        self.pdi = self.I(ta.PLUS_DI, self.high, self.low, self.close)
        self.mdi = self.I(ta.MINUS_DI, self.high, self.low, self.close)

    def next(self):
        if self.adx[-1] > 25 and self.adx[-2] <= 25 and self.pdi[-1] > self.mdi[-1] and not self.position.is_long:
            self.buy(size=0.2)
        if self.adx[-1] > 25 and self.adx[-2] <= 25 and self.mdi[-1] > self.pdi[-1] and not self.position.is_short:
            self.sell(size=0.2)
        # close buy position when price moves below PSAR and close sell position when price moves below psar
        if self.position.is_long and self.sar[-1] > self.close[-1] and self.sar[-2] <= self.close[-2]:
            self.position.close()
        if self.position.is_short and self.sar[-1] < self.close[-1] and self.sar[-2] >= self.close[-2]:
            self.position.close()



bt = Backtest(df, ParabolicStrategy, cash=100_000, commission=.002)
stats = bt.run()
print(stats)

#MISCELLANEOUS
# print(len(data_pd['chart'][0]))
# print(len(data_pd['indicator'][0][0]))
#
# # DIRECTIONAL INDEXES
# data_dis = open("data_di.json", "r")
# a_dictionary_dis = json.load(data_dis)
# data_pd_dis = pd.json_normalize(a_dictionary_dis)
# print("DATAPDDIS", len(data_pd_dis['indicator'][0][1]))
#
# df = pandas.DataFrame(columns=['adx','plus_di','minus_di'])
# # print(data_pd['chart'][0][0])
# # df_dict = pd.DataFrame(pd.Series([list(data_pd['chart'][0][0].items())]))
# # print(df_dict)
# print("FIRSTLEN IS", len(data_pd['indicator'][0][0]))
# print("SECONDLEN IS", len(data_pd_dis['indicator'][0][0]))
#
# for key in data_pd['chart'][0][0]:
#     print(key)
#     df[key] = pd.NaT
# #
# for ind, item in enumerate(data_pd['chart'][0]):
#     adx = []
#     adx.append(data_pd['indicator'][0][0][ind])
#     print("PLUSDI IS", data_pd_dis['indicator'][0][0][ind])
#     print("MINUSDI IS", data_pd_dis['indicator'][0][1][ind])
#     adx.append(data_pd_dis['indicator'][0][0][ind])
#     adx.append(data_pd_dis['indicator'][0][1][ind])
#     #print("ADX IS", adx)
#     #print("IND IS", ind)
#     values = []
#     for c in df.columns:
#         print("C IS", c)
#         if c != 'adx' and c != 'plus_di' and c!= 'minus_di':
#             values.append(data_pd['chart'][0][ind].get(c))
#             #print("VALUES IS", values)
#     adx.extend(values)
#     print(len(adx))
#     print(len(df.columns))
#     df.loc[len(df)] = adx
#
# configure_full_df_print()
# df.dropna(inplace=True)
# print(df)
# df2 = df[['adx','plus_di','minus_di','date','open','high','low','close','volume']]
# print(df2)
# aapl = df2
#
# ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
# ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
# ax1.plot(aapl['close'], linewidth = 2, color = '#ff9800')
# ax1.set_title('AAPL CLOSING PRICE')
# ax2.plot(aapl['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
# ax2.plot(aapl['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
# ax2.plot(aapl['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
# ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
# ax2.legend()
# ax2.set_title('AAPL ADX 14')
# plt.show()
#
#
# def implement_adx_strategy(prices, pdi, ndi, adx):
#     buy_price = []
#     sell_price = []
#     adx_signal = []
#     signal = 0
#
#     for i in range(len(prices)):
#         if adx.iloc[i - 1] < 25 and adx.iloc[i] > 25 and pdi.iloc[i] > ndi.iloc[i]:
#             if signal != 1:
#                 buy_price.append(prices.iloc[i])
#                 sell_price.append(np.nan)
#                 signal = 1
#                 adx_signal.append(signal)
#             else:
#                 buy_price.append(np.nan)
#                 sell_price.append(np.nan)
#                 adx_signal.append(0)
#         elif adx.iloc[i - 1] < 25 and adx.iloc[i] > 25 and ndi.iloc[i] > pdi.iloc[i]:
#             if signal != -1:
#                 buy_price.append(np.nan)
#                 sell_price.append(prices.iloc[i])
#                 signal = -1
#                 adx_signal.append(signal)
#             else:
#                 buy_price.append(np.nan)
#                 sell_price.append(np.nan)
#                 adx_signal.append(0)
#         else:
#             buy_price.append(np.nan)
#             sell_price.append(np.nan)
#             adx_signal.append(0)
#
#     return buy_price, sell_price, adx_signal
#
#
# buy_price, sell_price, adx_signal = implement_adx_strategy(aapl['close'], aapl['plus_di'], aapl['minus_di'], aapl['adx'])
#
# ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
# ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
# ax1.plot(aapl['close'], linewidth = 3, color = '#ff9800', alpha = 0.6)
# ax1.set_title('AAPL CLOSING PRICE')
# print(aapl.index)
# ax1.plot(aapl.index, buy_price, marker = '^', color = '#26a69a', markersize = 14, linewidth = 0, label = 'BUY SIGNAL')
# ax1.plot(aapl.index, sell_price, marker = 'v', color = '#f44336', markersize = 14, linewidth = 0, label = 'SELL SIGNAL')
# ax2.plot(aapl['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
# ax2.plot(aapl['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
# ax2.plot(aapl['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
# ax2.axhline(25, color = 'grey', linewidth = 2, linestyle = '--')
# ax2.legend()
# ax2.set_title('AAPL ADX 14')
# plt.show()
#
# # HOLD POSITION
# position = []
# for i in range(len(adx_signal)):
#     if adx_signal[i] > 1:
#         position.append(0)
#     else:
#         position.append(1)
#
# for i in range(len(aapl['close'])):
#     if adx_signal[i] == 1:
#         position[i] = 1
#     elif adx_signal[i] == -1:
#         position[i] = 0
#     else:
#         position[i] = position[i - 1]
#
# close_price = aapl['close']
# plus_di = aapl['plus_di']
# minus_di = aapl['minus_di']
# adx = aapl['adx']
# adx_signal = pd.DataFrame(adx_signal).rename(columns={0: 'adx_signal'}).set_index(aapl.index)
# position = pd.DataFrame(position).rename(columns={0: 'adx_position'}).set_index(aapl.index)
#
# frames = [close_price, plus_di, minus_di, adx, adx_signal, position]
# strategy = pd.concat(frames, join='inner', axis=1)
#
# print(strategy)
#
# ## BACKTEST
# aapl_ret = pd.DataFrame(np.diff(aapl['close'])).rename(columns={0: 'returns'})
# adx_strategy_ret = []
#
# for i in range(len(aapl_ret)):
#     returns = aapl_ret['returns'].iloc[i] * strategy['adx_position'].iloc[i]
#     adx_strategy_ret.append(returns)
#
# adx_strategy_ret_df = pd.DataFrame(adx_strategy_ret).rename(columns={0: 'adx_returns'})
# investment_value = 100000
# number_of_stocks = floor(investment_value / aapl['close'].iloc[-1])
# adx_investment_ret = []
#
# for i in range(len(adx_strategy_ret_df['adx_returns'])):
#     returns = number_of_stocks * adx_strategy_ret_df['adx_returns'][i]
#     adx_investment_ret.append(returns)
#
# adx_investment_ret_df = pd.DataFrame(adx_investment_ret).rename(columns={0: 'investment_returns'})
# total_investment_ret = round(sum(adx_investment_ret_df['investment_returns']), 2)
# profit_percentage = floor((total_investment_ret / investment_value) * 100)
# print(cl('Profit gained from the ADX strategy by investing $100k in AAPL : {}'.format(total_investment_ret),
#          attrs=['bold']))
# print(cl('Profit percentage of the ADX strategy : {}%'.format(profit_percentage), attrs=['bold']))


#a_file = open("data.json", "w")
#json.dump(data, a_file)

# Fetch marketcap from first endpont_path (no field_param used)
#print(f'Market Cap: {data["marketcap"]}')
