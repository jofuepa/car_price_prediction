from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def prepare_X(df, dv):
    df = df.copy()
    #fill missing values with zero
    df_num = df.fillna(0)
    
    #convert DataFrame to a Numpy array
    X = dv.transform(df_num.to_dict(orient='records'))
    return X

#Linear regression with normal equation with regularization
def train_linear_regression(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    #normal equation formula
    XTX = X.T.dot(X)
    #regularization
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

#Logistic regression
def train_sklearn_linear_regression (X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

def predict_sklearn(df, dv, model):
    X = prepare_X(df, dv)
    y_pred = model.predict(X)
    return y_pred

df = pd.read_csv('../../data/data_processed.csv')

car_dict = df.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(car_dict)

n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val+n_test)

#The seed is fixed to make sure that the results are reproducible
np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

#Shuffling the Dataframe with the help of idx

df_shuffled = df.iloc[idx]

#Splitting the shuffled Dataframe
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train = np.log1p(df_train.msrp.values)
y_val =  np.log1p(df_val.msrp.values)
y_test =  np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

X_train = prepare_X(df_train, dv)
model = train_sklearn_linear_regression(X_train, y_train)
y_predict = model.predict(X_train)

print('Training RMSE: ', rmse(y_train, y_predict))

y_predict = predict_sklearn(df_val, dv, model)
print('Validation RMSE: ', rmse(y_val, y_predict))

y_predict = predict_sklearn(df_test, dv, model)
print('Test RMSE: ', rmse(y_test, y_predict))

import pickle
with open('../../models/car-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)


