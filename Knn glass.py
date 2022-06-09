# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 08:50:44 2022

@author: varun
"""

import pandas as pd
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\KNN\\glass.csv")
df
list(df)
df.shape
df.head()
Y = df['Type']
Y
X = df.iloc[:,1:8]
X
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)
pd.crosstab(Y,Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale, Y, stratify=Y ,random_state=42 )

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)# k=25 #p = 2
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
Y_pred

from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, Y_pred)
print(cm)

import numpy as np
print(np.mean(Y_pred == Y_test).round(2))
print('Accuracy of KNN with K=25, on the test set: {:.3f}'.format(knn.score(X_test, Y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred).round(3)
