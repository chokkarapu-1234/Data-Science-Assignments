# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:54:43 2022

@author: varun
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\KNN\\Zoo.csv")
df
list(df)

x = df.iloc[:,1:17]
x
y = df['type']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, stratify = y, random_state = 42)
pd.crosstab(y,y)
#pip install KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) # k =5 # p=2 --> Eucledian distance
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(y_pred)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred == y_test).round(2))  
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(knn.score(x_test, y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

knn.score(x_test, y_test)
