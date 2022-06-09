# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:11:55 2022

@author: varun
"""

import pandas as pd
df1 = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Hypothesis Testing\\Cutlets.csv")
df1
df1.shape
list[df1]
df1.describe()
'''
#Test of Hypothesis
Ho: UnitA = UnitB ---> No significant difference in diameter of cutlets of two units
H1: UnitA != UnitB ---> Significant difference in diameter of cutlets of two units
'''

A = df1['Unit A']
B = df1['Unit B']
alpha = 0.05
from scipy import stats
ztest,pval =stats.ttest_ind(A,B)

print("Zcaluclated value is", ztest.round(4))
print("p-value value is", pval.round(4))
print(ztest,pval)
 
if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#Inference: As we have got to accept Ho that implies there is No significant
#---------- difference between the diameters of Unit A and Unit B
