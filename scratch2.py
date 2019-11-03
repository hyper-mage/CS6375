# https://www.datacamp.com/community/tutorials/lstm-python-stock-market

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(1)

# load the dataset
dataframe = pandas.read_csv('DJI.csv', usecols=['Date','Open','High','Low','Close'], engine='python')
# dataset = dataframe.values
# dataset = dataset[:,1:].astype('float32')

dataset = dataframe.sort_values('Date')
print(dataset.head())

plt.figure(figsize = (18,9))
plt.plot(range(dataset.shape[0]),(dataset['Low']+dataset['High'])//2)
plt.xticks(range(0,dataset.shape[0],500),dataset['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()
