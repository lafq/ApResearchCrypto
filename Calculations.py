import pandas as pd
import numpy as np
import warnings


class Calculations:
    # number of days in preferred time frame
    period = 14

    # TODO slice is not inclusive, so when separating for test use 12/31

    def __init__(self, filepath):
        # create new dataframe using given filepath
        self.df = pd.read_csv(filepath)
        warnings.filterwarnings("ignore")

        # change and use the date columns as the index
        self.df.index = pd.to_datetime(self.df['Date'])

        # drop original date column, and delete unnecessary info
        self.df = self.df.drop(['Date'], axis='columns')
        self.df = self.df.drop(['Adj Close'], axis='columns')

        self.df = self.create_new_variables()

    def create_new_variables(self):
        # simple variables
        self.df['High-Low'] = self.df.High - self.df.Low
        self.df['Open-Close'] = self.df.Open - self.df.Close

        # harder variables
        rsi = self.create_RSI()
        ema = self.create_EMA()
        mfi = self.create_MFI()

        # create new df with correct time frame and new variables
        new_df = self.df[self.period:]
        new_df['RSI'] = rsi
        new_df['EMA'] = ema
        new_df['MFI'] = mfi

        return new_df

    def create_RSI(self):
        change = self.df['Close'].diff()
        change.dropna(inplace=True)

        change_up = change.copy()
        change_down = change.copy()

        change_up[change_up < 0] = 0
        change_down[change_down > 0] = 0

        change.equals(change_up + change_down)

        avg_up = change_up.rolling(self.period).mean()
        avg_down = change_down.rolling(self.period).mean()

        rsi = 100 * avg_up / (avg_up + avg_down)

        return rsi[self.period:]

    def create_EMA(self):
        ema = self.df['Close'].ewm(com=0.8).mean()
        return ema[self.period:]

    def create_MFI(self):
        typical_price = (self.df['Close'] + self.df['High'] + self.df['Low']) / 3

        money_flow = typical_price * self.df['Volume']

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

        for i in range(self.period - 1, len(positive_flow)):
            positive_mf.append(sum(positive_flow[i + 1 - self.period: i + 1]))

        for i in range(self.period - 1, len(negative_flow)):
            negative_mf.append(sum(negative_flow[i + 1 - self.period: i + 1]))

        MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
        return MFI


"""
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
"""
