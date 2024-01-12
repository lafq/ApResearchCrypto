from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# plt.style.use('seaborn-dark grid')
warnings.filterwarnings("ignore")

df = pd.read_csv('Data/BTC-USD_daily.csv')

# changes the date columns as index
df.index = pd.to_datetime(df['Date'])

# delete original date column and delete unnecessary
df = df.drop(['Date'], axis='columns')
df = df.drop(['Adj Close'], axis='columns')

# create new variables
df['High-Low'] = df.High - df.Low
df['Open-Close'] = df.Open - df.Close

# RSI
change = df['Close'].diff()
change.dropna(inplace=True)

change_up = change.copy()
change_down = change.copy()

change_up[change_up < 0] = 0
change_down[change_down > 0] = 0

change.equals(change_up + change_down)

avg_up = change_up.rolling(14).mean()
avg_down = change_down.rolling(14).mean()

rsi = 100 * avg_up / (avg_up + avg_down)

# EMA

# MFI

split_percentage = 0.8
split = int(split_percentage * len(df))

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 20)

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)

ax1.plot(df['Close'], linewidth=2)
ax1.set_title('Bitcoin Close Price')

ax2.set_title('Relative Strength Index')
ax2.plot(rsi, color='orange', linewidth=1)

plt.show()
