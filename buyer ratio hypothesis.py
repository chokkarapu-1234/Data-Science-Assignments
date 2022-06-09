# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:44:47 2022

@author: varun
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Hypothesis Testing\\BuyerRatio.csv")
df

X1 = 50+142+131+70
X2 = 435+1523+1356+750
print(X1)
'''
H0:  both male and female buying ratio are same
H1: both male and female buying ratio are not equal 
'''
pme = 50/X1
pfe = 435/X2
from statsmodels.stats.proportion import proportions_ztest
var1 = np.array([pme,pfe])
var2=np.array([X1,X2])
stats,pval = proportions_ztest(var1,var2)

alpha = 0.05
if pval > alpha:
    print("accept h0, reject h1")
else:
    print("accept h1, reject h0")

pmw = 142/X1
pfw = 1532/X2
var3 = np.array([pmw,pfw])
var4 = np.array([X1,X2])
stats1,pval1 = proportions_ztest(var3,var4)
print(pval1)
print(stats1)
alpha = 0.05
if pval1 > alpha:
    print("accept h0, reject h1")
else:
    print("accept h1, reject h0")

pmn = 131/X1
pfn = 1358/X2
var5 = np.array([pmn,pfn])
var6 = np.array([X1,X2])
stats2,pval2 = proportions_ztest(var5,var6)
print(stats2,pval2)
alpha = 0.05
if pval2 > alpha:
    print("accept h0, reject h1")
else:
    print("accept h1, reject h0")
    
pms = 70/X1   
pfs = 750/X2
var7 = np.array([pms,pfs])
var8 = np.array([X1,X2])
stats3,pval3 = proportions_ztest(var7,var8)
print(stats3,pval3)
alpha = 0.05
if pval3 > alpha:
    print("accept h0, reject h1")
else:
    print("accept h1, reject h0")
    
