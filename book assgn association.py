# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:29:59 2022

@author: varun
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules # importing apriori , association rules 
from mlxtend.preprocessing import TransactionEncoder
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Association Rules\\book.csv")
list(df)
df.head()

df1=pd.get_dummies(df) # deleting unwanted null or nan in data set
df1

frequent_itemsets = apriori(df1, min_support=0.1, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules 
rules.sort_values('lift',ascending = False)
rules.sort_values('lift',ascending = False)[0:20]
rules[rules.lift>1]

rules[['support','confidence']].hist() # hist of support ,confidence
rules[['support','confidence','lift']].hist() # hist of support, confidence ,lift 


import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,56,23,57,87] # ploting some X and Y variables
y = [99,86,87,88,111,113,114,156]

plt.scatter(rules['support'], rules['confidence'])
plt.show()

import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents') # scatter support and confidence for similar in forzenset
plt.show()

''' conclusion:
  As per Lower the Confidence level Higher the no. of rules.
Higher the Support, lower the no. of rules.'''
    
    