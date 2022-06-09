# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:49:50 2022

@author: varun
"""

import pandas as pd
import seaborn as sns
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Random Forest\\Fraud_check.csv")
df.shape 
list(df)
df.head()
df.dtypes
x1 = df['Taxable.Income']
x1.shape
df.drop(['Taxable.Income'],axis = 1,inplace = True)
df
df_new = pd.concat([df,x1],axis =1)
x= df_new.iloc[:,0:5]
x
from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=  LabelEncoder()
x['Undergrad_1'] = LE.fit_transform(x['Undergrad'])
x['Marital.Status_1'] = LE.fit_transform(x['Marital.Status'])
x['Urban_1'] = LE.fit_transform(x['Urban'])
list(x)
list(x)
x.shape
x.dtypes
x.drop(['Undergrad','Marital.Status','City.Population'],axis =1,inplace = True)
y = df_new['Taxable.Income']
x['City.Population'].hist()
x['Work.Experience'].hist()
sns.countplot(x = 'Undergrad',data = df_new )
sns.countplot(x = 'Marital.Status',data = df_new )
sns.countplot(x = 'Urban',data = df_new)

y= df_new['Taxable.Income']
y.shape
list(y)

'''
#As per the problem statement we are asked to convert this Y variable into categorical
#So, Differentiating the Y variable with respect to mean
# Taxable.Income less  than or equal to 30000 is categorised as Risky, otherwise good 
'''
#Converting y variable in to Categorical
y1 =[]
for i in range(0,600,1):
    if y.iloc[i,]<= 30000:
       print('Risky')
       y1.append('Risky')
    else:
        print('good')
        y1.append('good')

y_new = pd.DataFrame(y1)        
LE=  LabelEncoder()
y_new = LE.fit_transform(y_new)
#Preprocessing the data
from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=  LabelEncoder()
SS = StandardScaler()

x_scale = SS.fit_transform(x.iloc[:,3:6:])
x_scale
x_scale = pd.DataFrame(x_scale)
x_new = pd.concat([x_scale,x['Undergrad_1'],x['Marital.Status_1'],x['Urban_1']],axis =1)

#Splitting into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,stratify=y_new,test_size=0.25,random_state=37)
x_train.shape

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.5,n_estimators=500)
model=RFC.fit(x_train,y_train)
y_pred=model.predict(x_test)


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
    
TR_err=np.mean(tr_err)
TE_err=np.mean(t_err)
TR_err 
TE_err

import matplotlib.pyplot as plt
plt.plot(set1,tr_err,label='Training error')
plt.plot(set1,t_err,label='Test error')
plt.xlabel('No of features')
plt.ylabel('Error')
plt.title('Graph')
plt.show()
