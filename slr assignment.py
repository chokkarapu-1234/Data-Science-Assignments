# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:00:00 2022

@author: varun
"""

import  pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\Documents\\excelr assignments\Simple Linear Regression\\Salary_Data.csv")
df
type(df)
list(df)

X = df['YearsExperience']
X.shape
X.ndim
type(X)
X = X[:, np.newaxis]

Y = df['Salary']
Y.shape
Y.ndim

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred = model.predict(X)

 import matplotlib.pyplot as plt
plt.scatter(X, Y,  color = 'Black')
plt.plot(X, Y_pred, color = 'RED')
Y_error = Y-Y_pred
print(Y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse
from math import sqrt
RMSE = sqrt(mse)
print(RMSE)
