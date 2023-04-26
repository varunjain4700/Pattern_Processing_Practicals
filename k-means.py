# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing our Mall Dataset
dataset = pd.read_csv("data3.csv")

from sklearn.cluster import KMeans

X = dataset.iloc[:,[3,4]]
X.head()
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    iner = kmeans.inertia_
    wcss.append(iner)

plt.plot(range(1,10) , wcss)
plt.xlabel("Number of Cluster")
plt.ylabel("Interia/WCSS")
plt.show()

# Training our Model
kmeans = KMeans(n_clusters = 5)
dataset['Cluster'] = kmeans.fit_predict(X)
dataset.head()

#Renaming the column names for our ease
dataset.columns
dataset = dataset.rename(columns = {"Annual Income (k$)" : "Annual_Income",
                           "Spending Score (1-100)" : "Spending_Score"} )
dataset.head()

# Lets visualize the clusters for some information
plt.scatter(x = dataset.loc[dataset.Cluster == 0 , 'Annual_Income'].values , y = dataset.loc[dataset.Cluster == 0 , 'Spending_Score'].values , s = 100 , c = 'red' , label = "Cluster1" )
plt.scatter(x = dataset.loc[dataset.Cluster == 1 , 'Annual_Income'].values , y = dataset.loc[dataset.Cluster == 1 , 'Spending_Score'].values , s = 100 , c = 'blue' , label = "Cluster2" )
plt.scatter(x = dataset.loc[dataset.Cluster == 2 , 'Annual_Income'].values , y = dataset.loc[dataset.Cluster == 2 , 'Spending_Score'].values , s = 100 , c = 'green' , label = "Cluster3" )
plt.scatter(x = dataset.loc[dataset.Cluster == 3 , 'Annual_Income'].values , y = dataset.loc[dataset.Cluster == 3 , 'Spending_Score'].values , s = 100 , c = 'brown' , label = "Cluster4" )
plt.scatter(x = dataset.loc[dataset.Cluster == 4 , 'Annual_Income'].values , y = dataset.loc[dataset.Cluster == 4 , 'Spending_Score'].values , s = 100 , c = 'magenta' , label = "Cluster5")
plt.scatter(x = kmeans.cluster_centers_[:,0] , y = kmeans.cluster_centers_[:,1] , s = 400 , c = 'yellow')
plt.xlabel('Annual Income in K$')
plt.ylabel('Spending Score in 1-100')
plt.title("Clustering the customers on basis of their Income and Spending Score")
plt.legend()
plt.show()