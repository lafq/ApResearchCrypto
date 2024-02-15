import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from Calculations import Calculations
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class RF:
    # BCH = 0.9695,     LTC = 0.9330
    # BTC = 0.9521,     OMG = 0.7395
    # DASH = 0.8910,    XMR = 0.9769
    # EOS = 0.9566,     XRP = 0.9824
    # ETC = 0.7256,     ZEC = 0.7534
    # ETH = 0.4138
    split = 1447

    def __init__(self, filepath):
        self.calcs = Calculations(filepath)

        # Separation of data
        target_close = self.calcs.df['Close']
        params = self.calcs.df.drop(['Close'], axis='columns')

        x_train = params[1: self.split]
        y_train = target_close[1: self.split]

        x_test = params[self.split:]
        self.y_test = target_close[self.split:]

        # Do not need to scale training data for RF

        # find best parameters
        rf = RandomForestRegressor()

        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_features': ["auto", "sqrt", "log2"],
            'min_samples_split': [2, 4, 8],
            'bootstrap': [False],
            'max_depth': [None, 10, 20]
        }

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_

        # Use the best parameters to predict
        y_prediction = best_model.predict(x_test)

        # save score
        self.score = best_model.score(x_test, self.y_test)

        # Fit prediction value into date index dataframe
        self.prediction = pd.DataFrame(y_prediction)
        self.prediction.index = x_test.index

    def plot(self):
        plt.plot(self.y_test, label='Close', color='red')
        plt.plot(self.prediction, label='Predicted', color='blue')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

    def print_score(self):
        print(f'R-squared score: {self.score}')
