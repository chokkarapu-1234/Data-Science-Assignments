# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:12:34 2022

@author: varun
"""

import pandas as pd
x_train = pd.read_csv("C:\\Users\\varun\\Downloads\\SalaryData_Train.csv")
x_train.shape
list(x_train)
x_test = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Navie Bayes\\SalaryData_Test.csv")
x_test.shape
list(x_test)

x_test.isnull().sum()
x_train.isnull().sum()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for eachcolumn in range(0,14,1):
    x_train.iloc[:,eachcolumn]= LE.fit_transform(x_train.iloc[:,eachcolumn])
    
x_train    

from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()

for eachcolumn in range(0,14,1):
    x_test.iloc[:,eachcolumn] = LE1.fit_transform(x_test.iloc[:,eachcolumn])
  
x_test

x_train1 = x_train.iloc[:,:13]
x_train1.shape
list,x_train1
x_train1
y_train1 = x_train.iloc[:,13:]
y_train1

x_test1 = x_test.iloc[:,:13]
x_test1
y_test1 = x_test.iloc[:,13:]
y_test1

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(x_train1,y_train1)
y_pred = MNB.predict(x_test1)
y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test1,y_pred)
acc = accuracy_score(y_test1,y_pred).round(2)*100
acc
print("naive bayes model accuracy score:" , acc)

'''
With the navie bayes method the Acuracy score is 78%

'''