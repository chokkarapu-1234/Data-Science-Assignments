# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:37:47 2022

@author: varun
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Hypothesis Testing\\LabTAT.csv")
df
'''
#Test of Hypothesis
Ho: l1 = l2 = l3 = l4 ---> All laboratories avg TAT is same
H1: l1 != l2 != l3 != l4 ---> Significant difference avg TAT among 4 laboratories
'''
l1=df['Laboratory 1']
l2=df['Laboratory 2']
l3=df['Laboratory 3']
l4=df['Laboratory 4']
import scipy.stats as stats
z,p=stats.f_oneway(l1,l2,l3,l4)
print(z,p)
alpha=0.05
if p>alpha:
    print('Accept Ho and Reject H1')
else:
    print('Accept H1 and Reject Ho')
    
#Inference: As per the test of hypothesis we are getting to accept H1,
#---------- so, there is significant difference between average TAT of 4 laboratories    