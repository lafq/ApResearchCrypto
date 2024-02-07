import pandas as pd
from sklearn.model_selection import GridSearchCV

from Calculations import Calculations
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

BTC = 'Data/BTC-USD_daily.csv'
calcs = Calculations(BTC)
df = calcs.df

split = 1447

params = df.drop(['Close'], axis='columns')
target_close = df['Close']

x_train = params[1:split]
y_train = target_close[1:split]

x_test = params[split:]
y_test = target_close[split:]

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

best_svr = SVR(kernel='linear', C=10, epsilon=0.1)
best_svr.fit(x_train_scaled, y_train)


y_prediction = best_svr.predict(x_test_scaled)

prediction = pd.DataFrame(y_prediction)
prediction.index = x_test.index

plt.plot(y_test, color='red')
plt.plot(prediction, color='blue')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()
