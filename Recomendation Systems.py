# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:26:09 2022

@author: varun
"""

import pandas as pd
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Recomendation  Systems\\book.csv",encoding="ISO-8859-1")
df
df.shape
list(df)

df['User.ID']
df['Book.Title']
df['Book.Rating']

df.drop(df.columns[[0]],axis =1,inplace =True)
df.sort_values('User.ID')

len(df)
len(df['User.ID'].unique())
len(df['Book.Title'].unique())
len(df['Book.Rating'].unique())


df['Book.Rating'].value_counts()
df['Book.Rating'].hist()


user_df=df.pivot_table(index ='User.ID',columns ='Book.Title',values ='Book.Rating')
pd.crosstab(df['User.ID'],df['Book.Title'])
user_df
user_df.iloc[0]
user_df.iloc[200]


#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)

user_df

# from scipy.spatial.distance import cosine correlation
#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances( user_df.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_df.values,metric='correlation')
user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df

#Set the index and column names to user ids 
user_sim_df.index   = df['User.ID'].unique()
user_sim_df.columns =df['User.ID'].unique()

user_sim_df.iloc[0:5, 0:5]

import numpy as np
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:5]

df[(df['User.ID']==276729) | (df['User.ID']==276726)]

user_276729=df[df['User.ID']==276729]

user_276726=df[df['User.ID']==276726]


user_276726=df[df['User.ID']==276726]
user_276736=df[df['User.ID']==276736]


pd.merge(user_276726,user_276736,on='Book.Rating',how='outer')



