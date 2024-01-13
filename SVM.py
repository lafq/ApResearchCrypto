from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

period = 14
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

rsi = rsi[period:]

# EMA
ema = df['Close'].ewm(com=0.8).mean()
ema = ema[period:]

# MFI
typical_price = (df['Close'] + df['High'] + df['Low']) / 3

money_flow = typical_price * df['Volume']

positive_flow = []
negative_flow = []

# Loop through the typical price
for i in range(1, len(typical_price)):
    if typical_price[i] > typical_price[i - 1]:
        positive_flow.append(money_flow[i - 1])
        negative_flow.append(0)

    elif typical_price[i] < typical_price[i - 1]:
        negative_flow.append(money_flow[i - 1])
        positive_flow.append(0)

    else:
        positive_flow.append(0)
        negative_flow.append(0)

positive_mf = []
negative_mf = []

for i in range(period - 1, len(positive_flow)):
    positive_mf.append(sum(positive_flow[i + 1 - period: i + 1]))

for i in range(period - 1, len(negative_flow)):
    negative_mf.append(sum(negative_flow[i + 1 - period: i + 1]))

MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
mfi = pd.DataFrame()
mfi['MFI'] = MFI

new_df = df[period:]
new_df['RSI'] = rsi
new_df['EMA'] = ema
new_df['MFI'] = MFI

split_percentage = 0.8
split = int(split_percentage * len(df))


plt.plot(new_df['Close'], label="Stock Values", color="black")
plt.plot(new_df['EMA'], label="EMA", color="red")
plt.plot(new_df['MFI'], label="MFI", color="blue")
plt.plot(new_df['RSI'], label="RSI", color="green")
plt.xlabel("Date")
plt.ylabel("Value")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()

