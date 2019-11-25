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

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train:\n", X_train)

# List compre-way
this = np.array([train_data_scaled[i-60:i, 0] for i in range(60, len(data))])
that = np.array([train_data_scaled[i, 0] for i in range(60, len(data))])
# Reshape dat b
this = np.reshape(this, (this.shape[0], this.shape[1], 1))
# print("X_train == this", (X_train == this).all())
# print("y_train == that", (y_train == that).all()) # this checks out so we can just use the list comp


