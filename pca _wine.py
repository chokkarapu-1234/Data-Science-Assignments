# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:02:27 2022

@author: varun
"""

import pandas as pd
df =  pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\PCA\\wine.csv")
df.shape
list(df)
df.dtypes
df.head(5)
x = df.iloc[:,1:15:]
x

#Preproceesing the data
from sklearn.preprocessing import StandardScaler
sf = StandardScaler()
x_scale = sf.fit_transform(x)
x_scale
x_scaler  =pd.DataFrame(x_scale)
x_scaler
#Decompostion
from sklearn.decomposition import PCA
pca1=PCA(n_components=3)
pca = PCA(svd_solver = 'full')
pc=pca.fit_transform(x_scale)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
pc.shape
type(pc)

pc_new = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6', 'pc_7', 'pc_8','pc_9','pc_10','pc_11','pc_12', 'pc_13'])
pc_new.head()
pc_new.to_csv("D:\\advaith sir\\Decision Tree\\PCA MODEL\\pca.csv",header = True)


import seaborn as sns 
df1_new = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['pc_1','pc_2','pc_3','pc_4','pc_5','pc_6','pc_7','pc_8','pc_9','pc_10','pc_11','pc_12','pc_13']})
sns.barplot(x = 'PC' , y = 'var',data = df1_new,color ='blue')

#Since almost 66% of the data is covered in first 3 bars as shown in the bar plot, we try by taking first 3 columns as X (as mentioned in the problem)
#Decomposition
from sklearn.preprocessing import PCA
pcal = PCA(n_components = 3)
pc1 = pcal.fit_transform(x_scale)
pcal.explained_variance_ratio_
sum(pcal.explained_variance_ratio_)
pc1.shape
type(pc1)

pc1_new = pd.DataFrame(data =pc1, columns = ['pc_1','pc_2','pc_3'])
type(pc1_new)
pc1_new.head()
pc1_new("D:\\advaith sir\\Decision Tree\\PCA MODEL\\pca1.csv",header = True)

import seaborn as sns
from sklearn.cluster import KMeans 
data1_new = pd.DataFrame({'var':pcal.explained_variance_ratio_,'PC':['pc_1','pc_2','pc_3']})
sns.barplot(x='PC',y='var',data=data1_new,color='red')
x1=pc1_new.iloc[:,0:3]
clust =[]
for i in range(1,10,1):
    km=KMeans(n_clusters=i).fit(x1)
    km.inertia_
    clust.append(km.inertia_)
print(clust)

import matplotlib.pyplot as plt
plt.plot(range(1,10),clust)
plt.title("Elbow plot")
plt.xlabel('No of Cluster')
plt.ylabel('Cluster inertia values')
plt.show()

#From Elbow plot we can observe that Cluster Inertial value is appreciably less for number of clusters=3.
#Considering 3 clusters
km=KMeans(n_clusters=3,random_state=98).fit(x1)
Y_pred=km.predict(x1)
Y_pred=pd.DataFrame(Y_pred)
Y_pred.value_counts()
type(Y_pred)

c=km.cluster_centers_
c.shape
km.inertia_

%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x1.iloc[:, 0],x1.iloc[:, 1],x1.iloc[:, 2])
ax.scatter(c[:, 0], c[:, 1], c[:, 2],marker='*', c='Red', s=1000) # S is star size, c= * color
plt.show()

Y=df['Type']
Y.value_counts()
type(Y)
'''
Original dataset contains the type as 1,2,3 but here python has given the type as 0,1,2 under Y_pred
To compare between the actual Y variable and predicted Y variable we need to convert those values
'''
Y_pred_new=[]
for i in range(0,178,1):
    if Y_pred.iloc[i,0]==0:
        Y_pred_new.append(1)
    elif Y_pred.iloc[i,0]==1:
        Y_pred_new.append(2)
    else:
        Y_pred_new.append(3)
Y_pred_df=pd.DataFrame(Y_pred_new)
Y_pred_df.value_counts()

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y,Y_pred_df)
cm
acc=accuracy_score(Y,Y_pred_df)
acc

#Hierarchial Clustering
#Construction of Dendogram
x2=pc1_new.iloc[:,0:3].values
x2
type(X2)
list(x2)
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.title('Dendrogram')
dend=shc.dendrogram(shc.linkage(x2,method='complete'))

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
Y_pred1=ac.fit_predict(x2)
Y_pred1=pd.DataFrame(Y_pred1)
Y_pred1.shape
Y_pred1.value_counts()

plt.figure(figsize=(16,9))
plt.scatter(x2[:,0],x2[:,1],x2[:,2],c=Y_pred1,cmap='rainbow')

%matplotlib qt
plt.rcParams['figure.figsize']=(16,9)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x2[:, 0],x2[:, 1],x2[:, 2])
plt.show()

Y=df['Type']
Y.value_counts()
type(Y)

'''
Original dataset contains the type as 1,2,3 but here python has given the type as 0,1,2 under Y_pred
To compare between the actual Y variable and predicted Y (Y_pred1) variable we need to convert those values
'''
Y_pred_new1=[]
for i in range(0,178,1):
    if Y_pred1.iloc[i,0]==0:
        Y_pred_new1.append(1)
    elif Y_pred1.iloc[i,0]==1:
        Y_pred_new1.append(2)
    else:
        Y_pred_new1.append(3)
        
Y_pred_df1=pd.DataFrame(Y_pred_new1)
Y_pred_df1.value_counts()

from sklearn.metrics import confusion_matrix,accuracy_score
cm1=confusion_matrix(Y, Y_pred_df1)
cm1
acc1=accuracy_score(Y, Y_pred_df1)
acc1       
