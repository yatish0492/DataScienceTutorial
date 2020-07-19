'''
K-Means Algorithm
-----------------
'''


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing mall dataset. The data has details about each customer of mall and his spending score based on how much he has spends.
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/4_1_K_Means.csv')
X = dataset.iloc[:,[3,4]].values     # We have taken salary and spend score

#Using Elbow Method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
# 'max_iter' is the maximum number of iterations that can be run by k-means. Default value of it is 300.
# 'init' can have 'random' as also the value which means it will use normal k-means not the k-means++ which can cause inconsistency.
# 'n_init' is the number of times it chooses the centeroids and run the iteration based on 'max_iter' value. it's default value is 10.
#       this means, 10 experiments will be conducted and will take best of them.
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    # kmeans have function in library called 'inertia_' which basically computes wcss value.
plt.plot(range(1,11), wcss) # we give x axis value and y axis values as parameters to 'plot' here 'range(1,11) is x axis values and wcss is y axis values.
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('wcss value')
plt.show

# From the elbow chart we found that '5' is the optimal number of clusters.

#Applyging k-means now with 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10)
clusteredResult = kmeans.fit_predict(X) 
# 'clusteredResult' is a single column of values which are the cluster numbers of the corresponding customers.
df = pd.DataFrame(clusteredResult, columns=['ClusterNumber']) # Creating a data frame from clusteredResult
df1 = pd.concat([df,dataset['CustomerID']], axis=1) # Concatinating 'CulusterNumber' and 'CustomerID' column to know which customer belongs to which cluster


'''

    Visualizing the clusters in the chart
    
'''
# X[clusteredResult == 0, 0] --> Means, consider the column of 'X' with index=0 as the x axis value and take only 0th cluster only
# X[clusteredResult == 0, 1] --> Means, consider the column of 'X' with index=1 as the y axis value and take only 0th cluster only
# X[clusteredResult == 1, 0] --> Means, consider the column of 'X' with index=0 as the x axis value and take only 1st cluster only
# X[clusteredResult == 1, 1] --> Means, consider the column of 'X' with index=1 as the y axis value and take only 1st cluster only
# c --> This means color. that point will colored with the value given
# label --> This point will be labeled as the value given. 
plt.scatter(X[clusteredResult == 0, 0], X[clusteredResult == 0, 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[clusteredResult == 1, 0], X[clusteredResult == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[clusteredResult == 2, 0], X[clusteredResult == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(X[clusteredResult == 3, 0], X[clusteredResult == 3, 1], c = 'grey', label = 'Cluster 4')
plt.scatter(X[clusteredResult == 4, 0], X[clusteredResult == 4, 1], c = 'pink', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c = 'black', label = 'Centeroids', s = 300) # 's' is size of point in chart.
plt.title('Clusters of Customers')
plt.xlabel('Anual Income (k $)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()    # This will show the color and label as legends in the chart so that we get to know which color is which cluster by looking at chart.
plt.show()






    