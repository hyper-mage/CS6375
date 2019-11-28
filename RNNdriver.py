import RNNfunction as rnn

train_X, train_y, test_X, test_y = rnn.prepare('DJI.csv', header=True, testsize=0.5, preprocess=True, timesteps=1)

rnn.RNN(train_X, train_y, 1)
