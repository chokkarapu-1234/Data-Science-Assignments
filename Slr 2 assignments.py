# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:54:49 2022

@author: varun
"""
import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\Simple Linear Regression\\delivery_time.csv")
df
type(df)
list(df)
 
X = df['Sorting Time']
X.shape
X.ndim
type(X)
X= X[:, np.newaxis]
X.ndim

Y = df['Delivery Time']
Y.shape
Y.ndim

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred = model.predict(X)
Y_pred
import matplotlib.pyplot as plt
plt.scatter(X, Y, color= 'black')
plt.plot(X, Y_pred, color  = 'red')
plt.show()
Y_error = Y-Y_pred
print(Y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse
from math import sqrt
Rmse = sqrt(mse)
Rmse
