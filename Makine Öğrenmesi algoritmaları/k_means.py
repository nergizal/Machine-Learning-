#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 21:38:40 2025

@author: nergizalici
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,2:4]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters =i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,10),sonuclar)
plt.show()

kmeans= KMeans(n_clusters = 4, init= 'k-means++', random_state=123)
Y_tahmin=kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1],s=100,c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1],s=100,c='green')
plt.title('KMeans')
plt.show()
#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3 , affinity = 'euclidean',linkage= 'ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==1,1] ,s=100,c='red')
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1] ,s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1] ,s=100,c='green')
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==1,1] ,s=100,c='yellow')
plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
dendogram = sch.dendogram(sch.linkage(X,method='ward'))
plt.show()










