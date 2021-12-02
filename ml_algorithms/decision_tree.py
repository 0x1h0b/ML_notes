'''
Decison Tree Implementation from scratch.
Important points : 
        - create a tree ds to store data
        - how to make rules for split , remember those splits
By - 0x1h0b

'''

import numpy as np
import pandas as pd

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


# class Node for tree
class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,info_gain=None,entropy=None,value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.entropy = entropy
        self.value = value

# decision tree classifier class
class myDecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2,feature_list=[],target_list=[]):
        ''' constructor '''
        # initialize the root of the tree 
        self.root = None
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        # feature list and target lists
        self.feature_list = feature_list
        self.target_list = target_list
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # printing the current node metadata of tree
        print('\nLevel ',curr_depth)
        val,count = np.unique(Y,return_counts=True)
#         print(val,count)
#         print('Y:',Y)
#         print(dataset)
        val,count = [int(i) for i in val],[int(j) for j in count]
        for idx in range(len(val)):
#             print(int(uv),type(uv))
            print('Count of '+str(self.target_list[val[idx]])+' = '+str(count[idx]))
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # print the node parameters
                gain_ratio = best_split["info_gain"]/best_split["entropy_curr_node"]
                print('Current Entropy is = '+str(best_split["entropy_curr_node"]))
                print('Splitting on feature '+str(self.feature_list[best_split["feature_index"]])+' with gain ratio = '+str(gain_ratio))
                
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],left_subtree, right_subtree, best_split["info_gain"],best_split["entropy_curr_node"])
        
        # compute leaf node
        print('Current entropy is = 0.0')
        print('Reached leaf node')
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain, curr_entropy = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        best_split["entropy_curr_node"] = curr_entropy
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        entropy_parent = self.entropy(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = entropy_parent - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain,entropy_parent
    
    def entropy(self, y):
        ''' function to compute entropy '''
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
#         print('y =>',y)
        class_labels = np.unique(y)
#         print('here')
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def get_unique_counts(self,y):
        l = np.unique(y)
        print()
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        ''' function to train the tree '''
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)



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


# dataset loading
data = datasets.load_breast_cancer()
x = data.data
y = data.target.reshape(-1,1)
t_list = data.target_names
f_list = data.feature_names


# # iris dataset
# iris_data = pd.read_csv('iris.csv')
# iris_data['species'].value_counts()
# dd={
#     'Iris-setosa':0,
#     'Iris-versicolor':1,
#     'Iris-virginica':2
# }
# iris_data['species'].replace(dd,inplace=True)
# f_list = ['sepal_length','sepal_width','petal_length','petal_width']
# t_list = list(dd.keys())
# print(iris_data.info())
# x = iris_data.iloc[:, :-1].values
# y = iris_data.iloc[:, -1].values.reshape(-1,1)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)
clf = myDecisionTreeClassifier(min_samples_split=3, max_depth=3,feature_list = f_list,target_list=t_list)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

report(y_test,y_pred)
