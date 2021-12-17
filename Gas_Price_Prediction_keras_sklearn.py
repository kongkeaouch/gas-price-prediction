import autokeras as ak
import tensorflow as tf
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from google.colab import drive

drive.mount('/content/drive')
df1 = pd.read_csv('/content/drive/kongkea/gas_train.csv')
df1.head()
df1.head()
df1.isnull().sum()
df2 = df1.dropna()
df2.head()
plt.scatter(df2.Date, df2.Gas, color='red', label='Prices', s=50)
df3 = df2[df2.Gas < 160]
plt.scatter(df3.Date, df3.Gas, color='blue', label='Prices', s=50)
df4 = df3.reset_index()['Gas']
df4.head()
scaler = MinMaxScaler(feature_range=(0, 1))
df4 = scaler.fit_transform(np.array(df4).reshape(-1, 1))
df4
training_size = int(len(df4)*0.65)
test_size = len(df4)-training_size
train_data, test_data = df4[0:training_size,
                            :], df4[training_size:len(df4), :1]
train_data.shape
test_data.shape


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:i+time_step, 0]
        dataX.append(a)
        dataY.append(dataset[(i+time_step, 0)])
    return numpy.array(dataX), numpy.array(dataY)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_train
y_train
print(X_train.shape), print(y_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(
    X_test, ytest), epochs=100, batch_size=64, verbose=1)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
look_back = 100
trainPredictPlot = numpy.empty_like(df4)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
testPredictPlot = numpy.empty_like(df4)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+look_back*2+1:len(df4)-1, :] = test_predict
plt.plot(scaler.inverse_transform(df4))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
df_test = pd.read_csv('gas_test.csv')
df_test.head()
model.save('gas price prediction')
df = pd.read_csv('/content/drive/kongkea/gas_train.csv')
df.head()
df.head()
df2 = df.dropna()
df3 = df2[df2.Gas < 160]
df3.head()
df4 = df3.reset_index()['Gas']
df4.head()
model1 = ARIMA(df4.values, order=(5, 1, 0))
model_fit1 = model1.fit()
output1 = model_fit1.forecast(steps=30)
output1
reg = ak.StructuredDataRegressor(overwrite=True, max_trials=3)
reg.fit(x=X_train, y=y_train, verbose=0)
mae, _ = reg.evaluate(X_test, ytest, verbose=0)
yhat_test = reg.predict(X_test)
model = reg.export_model()
model.summary()
yhat_train = reg.predict(X_train)
train_predict = scaler.inverse_transform(yhat_train)
test_predict = scaler.inverse_transform(yhat_test)
mean_squared_error(ytest, yhat_test)
mean_squared_error(y_train, yhat_train)
