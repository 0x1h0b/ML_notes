'''

randomly chose k examples as intital centroids
while true:
    create k clusters by assigning each example to closest centroids
    compute k new centroids by averaging examples in each cluster
    if centroids did not change:
        break


error - min,sum(varioance of each cluster)

'''



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




# K means algo




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

print(X_test,Y_test)

