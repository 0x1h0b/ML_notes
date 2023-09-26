'''

K-means algorithm belongs to the category of prototype based clustering

prototype based clustering means that each cluster is represented by a prototype,
which can be the centroid (avg) of similar points with cont. features, or the
medoid (most frequent) in the case of categorical features.

randomly chose k examples as intital centroids
while true:
    1.assign each point/sample to the nearest centroid
    2.Move the centroids to the center of the samples that were assigned to it.
    3.if centroids did not change:
        break

to determine similarity between 2 points , we calculate squared euclidean distance.

sum of squared error (SSE) also called cluster inertia (here we have to remember data needs to be scaled)
SSE = sum_i ,sum_j { W_i_j*euclidean(x_i,U_j)}

U_j -> centroid of cluster j
w_i_j -> 1 if sample x_i is in cluster j or 0

or

error - min,sum(variance of each cluster)

'''



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# K means algo
def k_means_cluster(x_tr,x_te,y_tr,y_te):
    kmc = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,random_state=0,tol=1e-04)

    kmc.fit(x_tr)
    print(kmc.cluster_centers_)
    # print(x_tr.iloc[:,0])

    # plot the 3 clusters
    plt.scatter(x_tr[x_te == 0, 0], x_tr[x_te == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
    # plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='cluster 2')
    # plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')
    plt.scatter(kmc.cluster_centers_[:, 0], kmc.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
    plt.show()

    # visualizing clusters  ??? --- TO DO



# loading iris dataset
def load_iris():
    iris_data = pd.read_csv('./datasets/iris.csv')
    dd={
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    iris_data['species'].replace(dd,inplace=True)
    Y = iris_data['species']
    X = iris_data.drop(['species'],axis=1)

    return X,Y



# load iris dataset
X,Y = load_iris()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=13)

k_means_cluster(X_train,X_test,Y_train,Y_test)


# print(X_test,Y_test)

