# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:55:54 2022

@author: varun
"""

import pandas as pd
import numpy as np
df =  pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Multi Linear Regression\\50_Startups.csv")
df
df.shape
list(df)
df.dtypes
df.head()

df.corr()
Y= df['Profit']
#X= df['R&D Spend']
#X = df[['R&D Spend', 'Marketing Spend',]]
#X= df[['R&D Spend',  'Administration']]
#X= df[['R&D Spend',  'Administration', 'Marketing Spend']]
X= df[['Administration', 'Marketing Spend']]
#X= X[:,np.newaxis]

X.ndim

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,Y)
model.intercept_
Y_pred= model.predict(X)
Y_pred

import matplotlib.pyplot as plt
plt.plot (X, Y_pred, color= 'red' )
plt.show()
Y_error = Y-Y_pred
print(Y_pred)
Y.ndim
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse
from math import sqrt
Rmse = sqrt(mse)
Rmse

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)*100
r2.round(3)

import statsmodels.api as sma
X2= sma.add_constant(X)
lm2 = sma.OLS(Y,X2).fit()
lm2.summary()
