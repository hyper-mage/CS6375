#%% Importante

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#%% Data importante

data = pd.read_csv('DJI.csv', index_col="Date", parse_dates=True)
print(data.head())
print("data information:\n", data.info())
data['Open'].plot(figsize=(16, 6))
# plt.show()

train_data = pd.DataFrame(data['Open'])


#%% Clean out any empties

print(data.isna().any())  # we good mayne

#%% Feature Scaling Normalization

scaler = MinMaxScaler(feature_range = (0, 1))
train_data_scaled = scaler.fit_transform(train_data)

#%% Slap some timesteps in they

t = 20

# List compre-way
this = np.array([train_data_scaled[i-t:i, 0] for i in range(t, len(data))])
that = np.array([train_data_scaled[i, 0] for i in range(t, len(data))])
# Reshape dat b
this = np.reshape(this, (this.shape[0], this.shape[1], 1))
# print("X_train == this", (X_train == this).all())
# print("y_train == that", (y_train == that).all()) # this checks out so we can just use the list comp

#%% Construct the RNN architecture

rnn = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (this.shape[1], 1)))
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
rnn.fit(this, that, epochs = 5, batch_size = 128)

