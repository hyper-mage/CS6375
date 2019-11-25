#%% Importante

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

X_train = []
y_train = []
for i in range(60, len(data)):
    X_train.append(train_data_scaled[i-60:i, 0])
    y_train.append(train_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train:\n", X_train)
print("y_train:\n", y_train)

# List compre-way
this = [train_data_scaled[i-60:i, 0] for i in range(60, len(data))]
that = [train_data_scaled[i, 0] for i in range(60, len(data))]
print("this\n", this, "\nthat\n", that)
print("X_train == this", X_train == this)
print("y_train == that", y_train == that)

