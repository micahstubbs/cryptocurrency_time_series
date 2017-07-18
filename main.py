import numpy
import pandas as pd
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Fix numpy random seed for reproducibility
numpy.random.seed(7)

# Read in CSV
bitcoin_dataframe = pd.read_csv('./bitcoin_ticker.csv', engine='python')
bitcoin_content = bitcoin_dataframe.values

print(bitcoin_dataframe.axes)
print(bitcoin_content[:1])

# Rescale data using sklearn.preprocessing.MinMaxScaler ('normalize' := scale to 0 ~ 1)
# (LSTM => sensitive to data scale... especially when using sigmoid() or tanh() activation functions
scaler = MinMaxScaler(feature_range=(0,1))
