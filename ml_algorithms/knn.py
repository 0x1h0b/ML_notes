'''
KNN Implementation from scratch.
Important points : 
        - for every test point we calulate the eucliden distance of that test point with 
        rest of the training data -> sort it ascending order -> pick the first k training points ->
        for classification give the mode of those points output , regression output the mean of the output of those 
        points.

        - pros : versatile for both regression and classification , simple algo
        - cons : High memory requirement ,predicting stage is very slow ,Computationaly expensive

        - class myKNN : its fit and predict method are for classification type problem
                        have to make changes if we want to use it for regression problem


By - 0x1h0b

'''

from os import stat
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# knn class from scratch
class myKNN:
    def __init__(self,n_neighbors=5):
        self.n_neighbors=n_neighbors
        self.X = None
        self.Y = None
    
    def fit(self,x_train,y_train):
        self.X = x_train
        self.Y = y_train
        return self

    def _predict(self,row):
        distance = []
        for i in range(len(self.X)):
            np_x = np.array(self.X.iloc[i,:])
            np_y = np.array(row)
            dis_ = np.sum(np.square(np_x-np_y))
            distance.append([dis_,i])
        
        distance = list(sorted(distance))[:self.n_neighbors]
        target_val = list(self.Y.values)
        ans = [target_val[item[1]] for item in distance]
        # print(ans)
        return Counter(ans).most_common(1)[0][0]

    def predict(self,x_test):
        
        predictions = []
        for idx,_row in x_test.iterrows():
            ans = self._predict(_row)
            predictions.append(ans)
        return predictions



# fucntion to generate classification report
def report(y_test,y_pred):
    print('\n\n\t-:: CLASSIFICATION SUMMARY ::-')

    acc = np.sum(y_test==y_pred)/len(y_test)
    print('\nAccuracy on test data :- ',acc)
    # calculate confusion matrix , and follow up error/
    conf_matrix = confusion_matrix(y_test,y_pred)
    print('\nConfusion matrix :-\n',conf_matrix)

    # classification report:
    report = classification_report(y_test,y_pred)
    print('\nClassification report :-\n',report)


# function to load titanic dataset and do some basic preprocessing
def load_titanic():
    df = pd.read_csv('./datasets/dataset_titanic.csv')

    df['Sex'] = np.where(df['Sex']=='male',0,1)
    df['Embarked'].fillna('S',inplace=True)
    df['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True)
    df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

    for i in ['Pclass','Sex','Embarked']:
        df[i] = df[i].astype('category')
    
    df_train_dummy = pd.get_dummies(df[['Pclass','Embarked']],drop_first=True)
    df = pd.concat([df[['Sex','Age','SibSp','Parch','Fare','Survived']],df_train_dummy],axis=1)
    df['Age'].fillna(df['Age'].mean(),inplace=True)

    Y = df['Survived']
    X = df.drop(['Survived'],axis=1)

    return X,Y

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


# loading titanic dataset
X,Y = load_titanic()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=13)



print("\n\n============ sklearn_KNN ==============")


## sklearn KNN with k=5
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
report(Y_test,Y_pred)

print("\n\n================ myKNN ================")

## my knn algo from scratch
my_clf = myKNN()
my_clf.fit(X_train,Y_train)
Y_pred_ = my_clf.predict(X_test)
report(Y_test,Y_pred_)
