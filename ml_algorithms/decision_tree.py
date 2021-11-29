'''
Decison Tree Implementation from scratch.
Important points : 
        - create a tree ds to store data
        - how to make rules for split
        -  
By - 0x1h0b

'''


from collections import Counter
from numpy.core.defchararray import split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


'''
# custom train_test_split method
import random
test_size = 0.4
def train_test_split(data,test_size=test_size):  # data is total preprocessed dataframe
    m = data.shape[0]
    random.seed(10)
    if isinstance(test_size,float):
        test_size = round(test_size*m)

    index = data.index.tolist()
    test_index = random.sample(population=index,k=test_size)

    test_df = data.loc[test_index]
    train_df = data.drop(test_index)

    y_train,y_test = train_df['label'] , test_df['label'] # if label is the output column
    x_train,x_test = train_df.drop(['label']),test_df.drop(['label'])

    return x_train,x_test,y_train,y_test

'''

def entropy(y):
    value_count = np.bincount(y)
    probability = value_count/len(y)
    result = np.sum([p*np.log2(p) for p in probability if p>0])
    return -result


class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self,min_sample_splits=2,max_depth=100,n_feature = None):
        self.min_sample_splits = min_sample_splits
        self.max_depth = max_depth
        self.n_feature = n_feature
        self.root = None
    
    def fit(self,x,y):
        self.n_feature = x.shape[1] if not self.n_feature else min(self.n_feature,x.shape[1])
        self.root = self._grow_tree(x,y)

    def _grow_tree(self,x,y,depth=0):
        n_sample,n_feature = x.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels ==1 or n_sample<self.min_sample_splits:
            leaf_node = self._most_common_label(y)
            return Node(value=leaf_node)

        feature_index = np.random.choice(n_feature,self.n_feature,replace=False)

        best_feature , best_threshold = self._best_criteria(x,y,feature_index)
        left_idx,right_idx = self._split(x[:,best_feature],best_threshold)

        left = self._grow_tree(x[left_idx,:],y[left_idx],depth+1)
        right = self._grow_tree(x[right_idx,:],y[right_idx],depth+1)

        return Node(best_feature,best_threshold,left,right)

    
    def _best_criteria(self,x,y,feature_index):
        best_gain = -1
        split_idx ,split_threshold = None,None
        for idx in feature_index:
            x_col = x[:,idx]
            thresholds = np.unique(x_col)
            for thres in thresholds:
                gain = self._information_gain(y,x_col,thres)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = thres
        return split_idx,split_threshold

    def _information_gain(self,y,x_col,split_threshold):
        parent_entropy = entropy(y)
        left_idx, right_idx = self._split(x_col,split_threshold)

        if len(left_idx)==0 or len(right_idx)==0:
            return 0
        
        n=len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        e_l ,e_r = entropy(y[left_idx]),entropy(y[right_idx])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        return parent_entropy - child_entropy

    
    def _split(self,x_col,split_thres):
        left_idx = np.argwhere(x_col<=split_thres).flatten()
        right_idx = np.argwhere(x_col > split_thres).flatten()
        return left_idx,right_idx


    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

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



data = datasets.load_breast_cancer()
x = data.data
y = data.target


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
# clf = DecisionTree(max_depth=12)
# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

report(y_test,y_pred)


from sklearn.tree import export_graphviz
import pydotplus

xx = export_graphviz(clf,out_file=None)
xy = pydotplus.graph_from_dot_file(xx)
xy.write_pdf('demo.pdf')


