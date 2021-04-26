import pandas as pd
import numpy as np

df = pd.read_csv("xor.csv")
shuffle_df = df.sample(frac=1)
train_size = int(0.7 * len(df))
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

X,y = np.array(train_set.iloc[:,0:-1]),np.array(train_set.iloc[:,-1])
X_test, y_test = np.array(test_set.iloc[:,0:-1]), np.array(test_set.iloc[:,-1])
X,y = np.ndarray(shape=X.shape, dtype=float, buffer=X),np.ndarray(shape=y.shape, dtype=float, buffer=y)
X_test, y_test = np.ndarray(shape=X_test.shape, dtype=float, buffer=X_test), np.ndarray(shape=y_test.shape, dtype=float, buffer=y_test)

(_,num_inputs)=X.shape
print(X.shape)
print(y.shape)