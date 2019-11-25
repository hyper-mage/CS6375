#%% non jupyter version of TODO.py
# will still break into cell chunks to be used in jupyter
# for testing

import pandas as pd 
import numpy as np 

#%% Data import

# fix random seed for reproducibility
np.random.seed(1)
# load the dataset
dataframe = pd.read_csv('DJI.csv', usecols=['Open','High','Low','Close'], engine='python')
# dataframe = pd.read_csv('DJI.csv', usecols=[-0], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# dataset = dataframe.sort_values('Date')
print(dataset)
print(dataframe.head())

# TODO: add preprocess stuff here

#%% Definitions
# TODO: set up RNN stuff

def activation(self, x, activation):
	if activation == 'sigmoid':
		return self.sigmoid(x)
	if activation == 'ReLu':
		return self.ReLu(x)
	if activation == 'tanh':
		return self.tanh(x)

def derivative(self, x, activation):
	if activation == 'sigmoid':
		return self.sigmoid_d(x)
	if activation == 'ReLu':
		return self.ReLu_d(x)
	if activation == 'tanh':
		return self.tanh_d(x)

def tanh(self, x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def tanh_d(self, x):
	return 1 - x**2

def ReLu(self, x):
	return np.maximum(0, x)
def ReLu_d(self, x):
	x[x <= 0] = 0
	x[x > 0] = 1
	return x

def sigmoid(self, x):
	return 1 / (1 + np.exp(-x))
def sigmoid_d(self, x):
	return x*(1 - x)

# TODO: set up rnn so they can stack

# TODO: add LSTM units here

# TODO: set up metrics here