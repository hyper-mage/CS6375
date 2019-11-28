#%% Importante

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


np.random.seed(1)

#%% helper function for determining how far back we look in time
def time_stepper(data, timesteps=1):
	# we use -1 because initial index is 0
    X = np.array([data[i:i+timesteps, 0] for i in range(len(data)-timesteps-1)])
    y = np.array([data[i+timesteps, 0] for i in range(len(data)-timesteps-1)])
    # reshape X with the samples, timesteps, features
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # timesteps, X.shape[1]))
    return X, y

#%% prepares the data before modelling
def prepare(data = 'DJI.csv', header=True, testsize=0.5, preprocess=True, timesteps=1):
    df = pd.read_csv(data, index_col='Date', parse_dates=True)
    # Graph the original data
    print('Graph of', data, 'with', timesteps, 'days out\n')
    # df['Open'].plot(figsize=(20, 8))
    # plt.title("Full Stock Movement")
    # plt.ylabel('Stock Prices')
    # plt.legend()
    # plt.show()
    df = pd.DataFrame(df['Open'])
    if preprocess:
        #%% Clean out any empties
        df.isna().any()
        #%% Feature Scaling Normalization
        # scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df)
    split = int(len(df)*testsize)
    train, test = df_scaled[0:split, :], df_scaled[split+1:len(df), :]
    train_X, train_y = time_stepper(train, timesteps)
    test_X, test_y = time_stepper(test, timesteps)
    return train_X, train_y, test_X, test_y, df






class RNN:
    def __init__(self, data, data_train_X, data_train_y, data_test_X, data_test_y, layers=1, timesteps=1):

        #%% Construct the RNN architecture

        rnn = Sequential()
        print("shape of data_train_X", data_train_X.shape[1])
        # Adding the first LSTM layer and some Dropout regularisation
        # may need to add timesteps here
        rnn.add(LSTM(units=50, return_sequences=True, input_shape=(data_train_X.shape[1], timesteps)))
        rnn.add(Dropout(0.2))

        # Loop extra layers in
        for i in range(layers):
            print("Adding layer", i+1)
            rnn.add(LSTM(units=50, return_sequences=True))
            rnn.add(Dropout(0.2))

        # Adding the last LSTM layer and some Dropout regularisation
        rnn.add(LSTM(units=50))
        rnn.add(Dropout(0.2))

        # Adding the output layer
        rnn.add(Dense(units=1))

        # Compiling the RNN
        rnn.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN to the Training set
        rnn.fit(data_train_X, data_train_y, epochs=5, batch_size=128)

        # Predict
        # make predictions
        trainPredict = rnn.predict(data_train_X)
        testPredict = rnn.predict(data_test_X)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([data_train_y])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([data_test_y])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % trainScore)
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % testScore)

        # shift train predictions for plotting, not needed
        # trainPredictPlot = data  # np.empty_like(data)
        # trainPredictPlot['Open'] = np.nan
        # trainPredictPlot[timesteps:len(trainPredict) + timesteps, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (timesteps * 2) + (1-(timesteps-1)):len(data) - timesteps - 1, :] = testPredict
        # plot baseline and predictions
        # print("data\n", data)
        # print("train\n", trainPredictPlot)
        # print("test\n", testPredictPlot)
        # data['Open'].plot(figsize=(20, 8))  # Tried to use this to force x-axis index to match datetime
        # plt.plot(trainPredict)
        # plt.plot(testPredict)
        pd.DataFrame(scaler.inverse_transform(scaler.fit_transform(data))).plot(figsize=(20, 8))
        plt.plot(scaler.inverse_transform(scaler.fit_transform(data)), color='black', label='Real Stock Movement',)
        plt.plot(trainPredict, color='orange', label='Training Predictions')
        plt.plot(testPredictPlot, color='green', label='Testing Predictions')
        plt.xlabel('Time in days')
        plt.ylabel('Stock Prices')
        plt.legend()
        plt.show()

        #
        # # print(data_test_X.head())
        # # scaler = MinMaxScaler(feature_range=(0, 1))
        # # data_test_X_scaled = scaler.fit_transform(data_test_X)  # to be safe, also prevents the error
        # prediction = pd.DataFrame(scaler.inverse_transform(rnn.predict(data_test_X)))
        #
        # # Graph the results
        # # plt.plot(data_test_y, color='blue', label='True Stock Price')
        # # plt.plot(prediction, color='orange', label='Predicted Stock Price')
        # # plt.title('Prediction vs True Stock Prices')
        # # plt.xlabel('Date')
        # # plt.ylabel('Stock Prices')
        # # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(data))
        # plt.plot(prediction)
        # # plt.plot(testPredictPlot)
        # plt.legend()
        # plt.show()
        #
        #
        #
        #
