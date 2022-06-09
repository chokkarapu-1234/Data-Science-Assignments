# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:47:54 2022

@author: varun
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\\Downloads\\ToyotaCorolla.csv",encoding='latin1')
df
df.shape
list(df)
df.corr()
corr_table=df.corr().to_csv("D:\\Assignments solutions\\corr_table.csv",header=True)

Y= df['Price']
#X= df['Age_08_04']
#X = df[['Age_08_04','Weight','KM']]
#X = df[['Age_08_04','Weight']]
#X = df[['Age_08_04','Weight','KM','HP']]
#X = df[['Age_08_04','Weight','KM','HP','Quarterly_Tax']]
#X = df[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors']]
#X = df[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc']]
#X = df[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc','Gears']]
#X = df[['Weight','KM']]
#X = df[['Age_08_04','KM']]
#X = df[['Age_08_04','KM','HP']]
X.ndim
#X= X[:,np.newaxis]
X.ndim

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred = model.predict(X)
Y_pred
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)*100
r2.round(3)

import statsmodels.api as sma
x1 = sma.add_constant(X)
lm2 = sma.OLS(Y,x1).fit()
lm2.summary()
