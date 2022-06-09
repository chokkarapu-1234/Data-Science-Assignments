# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:56:31 2022

@author: varun
"""


#SVM on salary dataset
'''
As we are provided with 2 datasets as train and test
Using train data to fit the model and test data to predict
'''

import pandas as pd
x_train = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Support Vector Machine\\SalaryData_Train(1).csv")
x_train.shape
list(x_train)
x_train.dtypes

x_test_1 = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Support Vector Machine\\SalaryData_Test(1).csv")
x_test_1.shape
list(x_test_1)
import seaborn as sns

#Data pre processing
#Seperating X and Y from both train and test datasets
x_train_1=x_train.iloc[:,0:13]
x_train_1.dtypes
x_train_1.hist()

sns.countplot(x_train_1['workclass'])
sns.countplot(x_train_1['education'])
sns.countplot(x_train_1['maritalstatus'])
sns.countplot(x_train_1['relationship'])
sns.countplot(x_train_1['sex'])
sns.countplot(x_train_1['native'])
sns.countplot(x_train_1['race'])

#Data which require standardization
x1=x_train[['age','educationno','capitalgain','capitalloss','hoursperweek']]
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x1_scale=SS.fit_transform(x1)
x1_df=pd.DataFrame(x1_scale)
x1_df.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1,inplace=True)
x1_df


#Data which require Label encoding
x2=x_train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
x2.shape
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,8,1):
    x2.iloc[:,i]=LE.fit_transform(x2.iloc[:,i])
type(x2)
x_train_new=pd.concat([x1_df,x2],axis=1)
y_train=x_train.iloc[:,13:]
y_train
y_train.dtypes


y_train_scale=LE.fit_transform(y_train)
y_train_df=pd.DataFrame(y_train_scale)
list(y_train_df)
y_train_df.set_axis(['Salary'],axis='columns',inplace=True)
y_train_df.ndim
y_train_df['Salary'].ndim

x_test = x_test_1.iloc[:,0:13]
x_test
x_test.dtypes
x_test.hist()

#Standarization
x3 = x_test[['age','educationno','capitalgain','capitalloss','hoursperweek']]
ss = StandardScaler()
x3_scale = ss.fit_transform(x3)
x3_df = pd.DataFrame(x3_scale)
x3_df.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1,inplace=True)
x3_df

#Label Encoding
x4= x_test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
x4.shape
for i in range(0,8,1):
    x4.iloc[:,i] = LE.fit_transform(x4.iloc[:,i])
type(x4)    
x_test_new = pd.concat([x3_df,x4],axis =1)

y_test = x_test_1.iloc[:,13:]
y_test
y_test_scale = LE.fit_transform(y_test)
y_test_new = pd.DataFrame(y_test_scale)
y_test_new
y_test_new.set_axis(['Salary'],axis='columns',inplace=True)
y_test_new.ndim
y_test_new['Salary'].ndim


#Loading SVC (Linear kernel)
from sklearn.svm import SVC
svl=SVC(kernel='linear').fit(x_train_new,y_train_df['Salary'])
y_pred=svl.predict(x_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test_new['Salary'],y_pred)
print((acc*100).round(3))

#Loading SVC (Radial Bias Function kernel)
from sklearn.svm import SVC
svr=SVC(kernel='rbf').fit(x_train_new,y_train_df['Salary'])
y_pred=svr.predict(x_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_test_new['Salary'],y_pred)
print((acc1*100).round(3))

#Loading SVC (Polynomial kernel)
from sklearn.svm import SVC
svp=SVC(kernel='poly').fit(x_train_new,y_train_df)
y_pred=svr.predict(x_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc2=accuracy_score(y_test_new,y_pred)
print((acc2*100).round(3))

'''
Inference: From the different kernel functions in svm, the accuracies are almost same.
'''
