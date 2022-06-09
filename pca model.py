
"""
Created on Tue Apr 26 11:48:32 2022

@author: varun
"""

import pandas as pd
df = pd .read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\PCA\\wine.csv")
print(df)
df.shape

x = df.iloc[:,1:14]
print(x)
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
x_scaler = x_scale.fit_transform()
x_scaler
x_scale = pd.DataFrame(x_scaler)
x_scale

from sklearn.decomposition import PCA
varun = PCA(svd_solver = 'full')
varun

pc = varun.fit_transform(x_scale)
varun.explained_variance_ratio_
sum(varun.explained_variance_ratio_)

pc.shape
pd.DataFrame(pc).head(7)

import seaborn as sns
df = pd.DataFrame({'var':varun.explained_variance_ratio_,
                  'PC':['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13']})
            
sns.barplot(x = 'PC',y = "var",data = df,color ="c");
 
varun = PCA(n_components = 9)
varun_1 = varun.fit_transform(x_scale)
varun_1
varun_ = pd.DataFrame(data = varun_1 , columns =['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9'])
varun_
from sklearn.preprocessing import StandardScaler

varun.explained_variance_ratio_
import seaborn as sns
df = pd.DataFrame({'var':varun.explained_variance_ratio_,
                   'pc':['pc1','pc2','pc3']})
sns.barplot(x = 'pc',y = "var",data = df,color ="c");

x = x_scale
type(x)
x.shape
print(x)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= (16,9)
%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig =plt.figure()
ax = Axes3D(fig)
ax.scatter(x.iloc[:,0], x.iloc[:,1],x.iloc[:,2], x.iloc[:,3],x.iloc[:,4],x.iloc[:,5])
plt.show()
print(x)
x.dtypes
