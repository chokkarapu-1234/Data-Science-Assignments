# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:31:04 2022

@author: varun
"""

import pandas as pd
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Association Rules\\my_movies.csv")
df.shape
df.head()

df1 = df.iloc[:,5:]
df1.head()
df1.describe().T

df1.isnull().sum()
df1.dtypes

x ={}
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(df1).transform(df1)
x1 = pd.DataFrame(te_ary,columns = te.columns_)
import matplotlib as plt
x1.sum().to_frame('Frequency').sort_values('Frequency',ascending =False)[:260].plot(kind = 'bar',
                                                                                     figsize =(13,6),
                                                                                     title ="frequent items")
from mlxtend.frequent_patterns import apriori,association_rules

ap_0_5 = {}
ap_1 = {}
ap_5 = {}
ap_1_0 = {}

confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def gen_rules(df1,confidence,support):
    ap = {}
    for i in confidence:
        ap_i =apriori(df1,support,True)
        rule = association_rules(ap_i,min_threshold = i)
        ap[i]= len(rule.antecedents)
    return pd.Series(ap).to_frame("Support: %s"%support)    

confs = []

for i in [0.005,0.001,0.003,0.007]:
    ap_i = gen_rules(x1,confidence=confidence,support=i)
    confs.append(ap_i)
    
all_conf = pd.concat(confs,axis=1)
   
all_conf.plot(figsize=(8,8),grid=True)
plt.ylabel('Rules')
plt.xlabel('Confidence')
plt.show()    

'''
4 - Conclusiom
As shown In above graph

Lower the Confidence level Higher the no. of rules.
Higher the Support, lower the no. of rules.
'''

ap_final =  apriori(x1,0.005,True)
rules_final = association_rules(ap_final,min_threshold=.4,support_only=False)
rules_final[rules_final['confidence'] > 0.5]

support = rules_final["support"]
confidence =  rules_final["confidence"]
lift = rules_final["lift"]
from mpl_toolkits.mplot3d import Axes3D

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(support,confidence,lift)
ax1.set_xlabel("support")
ax1.set_ylabel("confidence")
ax1.set_zlabel("lift")

plt.scatter(support,confidence, c =lift, cmap = 'gray')
plt.colorbar()
plt.xlabel("support");plt.ylabel("confidence")


 ap_i = gen_rules(ap,confidence=confidence,support=i)
confs.append(ap_i)

