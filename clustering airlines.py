# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:41:03 2022

@author: varun
"""

import pandas as pd
df = pd.ExcelFile("C:\\Users\\varun\\Documents\\excelr assignments\\Clustering\\EastWestAirlines.xlsx")
df1 = pd.read_excel(df,sheet_name = "data")
df1
list(df1)
df1.drop(['ID#'],axis = 1,inplace =True)
df1.shape
df1.head()
x = df1.iloc[:,0:11]
x
x.shape
list(x)
import scipy.cluster.hierarchy as shc
#construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize =(10,7))
plt.title("x Dendogram")
dend = shc.dendrogram(shc.linkage(x, method = 'complete'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'complete' )
y = cluster.fit_predict(x)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
y_clust = pd .DataFrame(y)
y_clust[0].value_counts()

%matplotlib qt
plt.rcParams['figure.figsize'] =[16,9]
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.iloc[:, 0], x.iloc[:, 1], x.iloc[:, 2])
plt.show()

from sklearn.cluster import KMeans
Kmeans = ()
kmeans = KMeans(n_clusters = 3)
kmeans = kmeans.fit(x)
labels = kmeans.predict(x)
type(labels)
C = kmeans.cluster_centers_
kmeans.inertia_

%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.iloc[:, 0], x.iloc[:, 1], x.iloc[:, 2]) 
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='Red', s=1000)

y = pd.DataFrame(labels)
x

df_new = pd.concat([pd.DataFrame(x),y],axis =1)
pd.crosstab(y[0],y[0])
y

clust =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,random_state = 0)
    kmeans.fit(x)
    clust.append(kmeans.inertia_)
    
    
plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()    

print(clust)    


# DBSCAN 
from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(df1)
X = stscaler.transform(df1)
X
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=2, min_samples=6)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()

clustered = pd.concat([df1,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered

a=0
while a<5:
  print(a)
  a=a+1
clustered.mean()
finaldata.mean()