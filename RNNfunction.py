#%% Importante

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(1)

#%% helper function for determining how far back we look in time
def time_stepper(data, timesteps=1):
    X = np.array([data[i-timesteps:i, 0] for i in range(timesteps, len(data))])
    y = np.array([data[i, 0] for i in range(timesteps, len(data))])
    # reshape X with the samples, timesteps, features=1('Open')
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

#%% prepares the data before modelling
def prepare(data = 'DJI.csv', header=True, testsize=0.5, preprocess=True, timesteps=1):
    df = pd.read_csv(data, index_col='Date', parse_dates=True)
    print('Graph of', data, 'with', timesteps, 'days out\n')
    df['Open'].plot(figsize=(20,8))
    plt.show()
    df = pd.DataFrame(df['Open'])
    if preprocess:
        #%% Clean out any empties
        df.isna().any()
        #%% Feature Scaling Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df)
    split = int(len(df)*testsize)
    train, test = df_scaled[0:split, :], df_scaled[split+1:len(df), :]
    train_X, train_y = time_stepper(train, timesteps)
    test_X, test_y = time_stepper(test, timesteps)
    return train_X, train_y, test_X, test_y






class RNN:
    def __init__(self, data_X, data_y, timesteps):

        #%% Construct the RNN architecture

        rnn = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (data_X.shape[1], timesteps)))
        rnn.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        rnn.add(LSTM(units = 50, return_sequences = True))
        rnn.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        rnn.add(LSTM(units = 50, return_sequences = True))
        rnn.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        rnn.add(LSTM(units = 50))
        rnn.add(Dropout(0.2))

        # Adding the output layer
        rnn.add(Dense(units = 1))

        # Compiling the RNN
        rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Fitting the RNN to the Training set
        rnn.fit(data_X, data_y, epochs = 5, batch_size = 128)



