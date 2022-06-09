# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:32:17 2022

@author: varun
"""

import pandas as pd
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Random Forest\\Company_Data.csv")
df.shape
list(df)
df.shape
df.dtypes
x1 = df['ShelveLoc']
x1.head
df.drop(['ShelveLoc'],axis =1,inplace =True)
df_new = pd.concat([df,x1],axis =1)
df_new
list(df_new)


list(x1)
x = df_new.iloc[:,1:11:]
x
list(x)
x.iloc[:,0:7].hist()
import seaborn as sns
sns.countplot(x = 'Urban',data = df)
sns.countplot(x='US', data = df_new )
sns.countplot(x ='ShelveLoc', data = df_new)

list(x1)
x1

y = df_new['Sales']
list(y)
y.shape
y_mean = y.mean()
y_mean

'''
#As per the problem statement we are asked to convert this Y variable into categorical
#So, Differentiatcng the Y variable with respect to mean
#Sales greater than or equal to mean is categorised as High, otherwise Low
'''

#Converting Y variable into categorical
y1 = []
for i in range(0,400,1):
        if y.iloc[i,]>=y_mean:
            print('high')
            y1.append('high')
        else:
            print('low')
            y1.append('low')
y_new=LE.fit_transform(y_new)
y_new = pd.DataFrame(y1)            
y_new.set_axis(['sales'],axis = 'columns')           

from  sklearn.preprocessing import LabelEncoder,StandardScaler
LE = LabelEncoder()
SS = StandardScaler()
x['Urban'] = LE.fit_transform(x['Urban'])
x['US'] = LE.fit_transform(x['US'])
x['ShelveLoc'] = LE.fit_transform(x['ShelveLoc'])
x_scale = SS.fit_transform(x.iloc[:,0:6:])
list(x_scale)
x_scale = pd.DataFrame(x_scale)
x_scale
x_new = pd.concat([x_scale,x['Urban'],x['US'],x['ShelveLoc']],axis =1)
x_new

#Pre processing the data to  get training and test error
from sklearn.model_selection import train_test_split
y_new=LE.fit_transform(y_new)
x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,stratify=y_new,test_size=0.25,random_state=37)
x_train.shape

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.4,n_estimators=500)
model=RFC.fit(x_train,y_train)
Y_pred=model.predict(x_test)

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
acc=accuracy_score(y_test, y_pred)
print(acc)

import numpy as np
tr_err=[]
t_err=[]
set1=np.arange(0.1,1.1,0.1)
for j in set1:
    RFC=RandomForestClassifier(max_features=j,n_estimators=500)
    model=RFC.fit(x_train,y_train)
        
    y_pred_tr=model.predict(x_train)
    y_pred_te=model.predict(x_test)
        
    tr_err.append(np.sqrt(metrics.mean_squared_error(y_train,y_pred_tr)))
    t_err.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred_te)))
    
import matplotlib.pyplot as plt
plt.plot(set1,tr_err,label='Training error')
plt.plot(set1,t_err,label='Test error')
plt.xlabel('No of features')
plt.ylabel('Error')
plt.title('Graph')
plt.show()    
'''
From the graph we can observe that for max_features of 0.4 i.e, considering the 40% of the columns
we can see the minimum test error, so taking it to fit the model
'''      
