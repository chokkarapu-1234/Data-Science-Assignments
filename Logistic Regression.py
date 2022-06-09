# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:58:26 2022

@author: varun
"""

import pandas as p
df = p.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Logistic Regression\\bank-full (1).csv" ,sep =";")
df
df.shape
list(df)
x = df.iloc[:,0:16:]
x
x1 =df[['job','marital','education','default','housing','loan','contact','month','poutcome']]
x1.head()
x1.shape
type(x1)
y =df['y']
#Converting to Numeric format
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,9,1):
    x1.iloc[:,i] = LE.fit_transform(x1.iloc[:,i])
print(x1)

x2=df[['age','balance','day','duration','campaign','pdays','previous']]
x2.head()

from sklearn.preprocessing import StandardScaler
x2_new = StandardScaler().fit_transform(x2)
print(x2_new)

from sklearn.preprocessing import MinMaxScaler
x2_new2 = MinMaxScaler().fit_transform(x2)
print(x2_new2)

#Fitting the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(x2_new,y)
y_pred=model.predict(x2_new)
y_pred
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm=confusion_matrix(y,y_pred)
acc=accuracy_score(y,y_pred)
rec=recall_score(y,y_pred)
f1=f1_score(y,y_pred)
print(cm)
print(acc)

'''

