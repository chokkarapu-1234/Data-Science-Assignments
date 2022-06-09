# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:51:02 2022

@author: varun
"""

import pandas as pd 
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Desicion Tree\\Company_Data.csv")
df.shape
list(df)
df.head()

x1 = df['ShelveLoc']
df.drop(['ShelveLoc'],axis =1,inplace =True)
df
df_new = pd.concat([df,x1],axis =1)
df_new.head()

x = df_new.iloc[:,1:11]
x
x.iloc[:,0:7:].hist()
import seaborn as sns 
sns.countplot(x='Urban', data= df_new)
sns.countplot(x = 'US',data = df_new)
sns.countplot(x = 'ShelveLoc', data =df_new)

y = df_new['Sales']
y.shape
y_mean = y.mean()
y_mean

'''
#As per the problem statement we are asked to convert this Y variable into categorical
#So, Differentiatcng the Y variable with respect to mean
#Sales greater than or equal to mean is categorised as High, otherwise Low
'''
y1= []
for i in range(0,400,1):
    if y.iloc[i,]>=y_mean:
        print('high')
        y1.append('high')
    else:
        print('low')
        y1.append('low')
    
from sklearn.preprocessing import LabelEncoder,StandardScaler        
LE = LabelEncoder()
y_new = LE.fit_transform(y1)
y_new
y_new = pd.DataFrame(y_new)

SS = StandardScaler()
x['Urban'] = LE.fit_transform(x['Urban'])
x['US'] = LE.fit_transform(x['US'])
x['ShelveLoc'] =LE .fit_transform(x['ShelveLoc'])
x_scale = SS.fit_transform(x.iloc[:,0:6])
x_scale = pd.DataFrame(x_scale)
x_new = pd.concat([x_scale,x['Urban'],x['US'],x['ShelveLoc']],axis =1)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,stratify =y_new,test_size= 0.25,random_state= 62)

#Decision tree Classifier (As Y have 2 outputs we choose Classifier)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy',max_depth =8).fit(x_train,y_train)
y_pred = DT.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
acc

#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))
