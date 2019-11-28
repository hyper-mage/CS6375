import RNNfunction as rnn

# 1 day out
timesteps = 1
train_X, train_y, test_X, test_y, df = rnn.prepare('DJI.csv',
                                                   header=True, testsize=0.5, preprocess=True, timesteps=timesteps)
rnn.RNN(df, train_X, train_y, test_X, test_y, layers=2, timesteps=timesteps)

# 2 days out
timesteps = 2
train_X, train_y, test_X, test_y, df = rnn.prepare('DJI.csv',
                                                   header=True, testsize=0.5, preprocess=True, timesteps=timesteps)
rnn.RNN(df, train_X, train_y, test_X, test_y, layers=2, timesteps=timesteps)

# 3 days out
timesteps = 3
train_X, train_y, test_X, test_y, df = rnn.prepare('DJI.csv',
                                                   header=True, testsize=0.5, preprocess=True, timesteps=timesteps)
rnn.RNN(df, train_X, train_y, test_X, test_y, layers=2, timesteps=timesteps)

# 7 days out
timesteps = 7
train_X, train_y, test_X, test_y, df = rnn.prepare('DJI.csv',
                                                   header=True, testsize=0.5, preprocess=True, timesteps=timesteps)
rnn.RNN(df, train_X, train_y, test_X, test_y, layers=2, timesteps=timesteps)

# 30 days out
timesteps = 30
train_X, train_y, test_X, test_y, df = rnn.prepare('DJI.csv',
                                                   header=True, testsize=0.5, preprocess=True, timesteps=timesteps)
rnn.RNN(df, train_X, train_y, test_X, test_y, layers=2, timesteps=timesteps)
