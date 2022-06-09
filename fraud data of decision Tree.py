# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:31:51 2022

@author: varun
"""

import pandas as pd
import matplotlib as plt

df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Desicion Tree\\Fraud_check.csv")
df
df.shape
df.dtypes
list(df)
x1 = df['Taxable.Income']
x1.shape
df.drop(['Taxable.Income'],axis =1, inplace = True)
df
x2 =[]
for i in range(0,600,1):
    if x1.iloc[i,]<=30000:
        print('Risky')
        x2.append('Risky')
    else:
        print('Good')
        x2.append('Good')
x2        
x2_new = pd.DataFrame(x2)
x2_new
x2_new.set_axis(['Category'],axis='columns',inplace=True)
df_new = pd.concat([df,x2_new],axis =1)

#Splitting data into X and Y
x =  df_new.iloc[:,0:5:]
x.head()
y = df_new['Category']
y.head()

# preprocessing the data
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
x['Undergrad'] = LE.fit_transform(x['Undergrad'])
x['Marital.Status'] = LE .fit_transform(x['Marital.Status'])
x['Urban'] = LE.fit_transform(x['Urban'])
y = LE.fit_transform(y)
print(x)
print(y)

#spiliting the data in to train and test
from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25,stratify = y,random_state = 45 )
x_train.shape

#Decision tree Classifier (As Y have 2 outputs we choose Classifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
DT = DecisionTreeClassifier(criterion='entropy',max_depth =4).fit(x_train,y_train)
y_pred = DT.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print((acc*100).round(3))

#Tree
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth

DT = DecisionTreeClassifier(criterion='gini',max_depth =4).fit(x_train,y_train)
y_pred = DT.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print((acc*100).round(3))
