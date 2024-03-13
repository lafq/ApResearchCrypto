import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from Calculations import Calculations
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class SVM:
    # BCH = -11.5596,  LTC = -1.0933
    # BTC = -0.4567,   OMG = -4.3333
    # DASH = -19.0565, XMR = -2.3508
    # EOS = -33.4149,  XRP = 0.5120
    # ETC = -2.0739,   ZEC = -0.8221
    # ETH = -1.1381

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

        # Scale training data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)

        # Used to find best hyperparameters
        svr = SVR()

        hyperparameter_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.5]
        }

        grid_search = GridSearchCV(estimator=svr, param_grid=hyperparameter_grid, cv=5)
        grid_search.fit(x_train_scaled, y_train)

        best_params = grid_search.best_params_
        best_svr = SVR(**best_params)

        # Use the best hyperparameters to predict
        best_svr.fit(x_train_scaled, y_train)
        y_prediction = best_svr.predict(x_test_scaled)

        # R squared score of total dataset
        self.score = best_svr.score(x_test_scaled, self.y_test)

        # Fit prediction value into date index dataframe
        self.prediction = pd.DataFrame(y_prediction)
        self.prediction.index = x_test.index

    def print_score(self):
        print(f'R-squared score: {self.score}')

    def plot(self):
        plt.plot(self.y_test, label='Close', color='red')
        plt.plot(self.prediction, label='Predicted', color='blue')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

    def final_dataframe(self):
        df = pd.DataFrame()
        df['Close'] = self.y_test
        df['Predicted'] = self.prediction
        df.index = self.prediction.index
        return df
