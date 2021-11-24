
'''
Vectorized implementation of logistic regression using gradient descent algorithm.
Important points : 
        - fourmula of cost fucntion , partial derivatives , sigmoid function
        - make sure the vector dimensions are as expected before doing numpy operations 
        - Confusion matrix and classification report

By - 0x1h0b

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# calulate h(x,theta) , hypothesis value for logistic regression
def hypothesis(x,theta):
    h1 = np.dot(x,theta)
    return 1/(1+np.exp(-h1))

# calculate cost(x,theta,y,m)
def cost(x,theta,y):
    part1 = np.dot(y.T,np.log(hypothesis(x,theta)))
    part2 = np.dot((1-y).T,np.log(1-hypothesis(x,theta)))
    # print(part1+part2)
    return part1+part2

def gradient_descent(x,y, alpha, epoch):
    m,n = x.shape[0],x.shape[1] # no of training examples, no of features
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    theta = np.ones((n+1,1))
    n = n+1 # +1 cause we added the theta0 or corresponding x0 (intercept) to x
    cost_values=[]
    
    y = np.reshape(y.values,(m,1)) # previous shape (m,)
    # starting algo
    print('\n\n\tStarting gradient descent !!!')
    for it in range(epoch+1):

        # while calulating theta , trying to reduce a for loop by using vectors , partial derivaative in maths
        theta -= alpha*(1/m)*np.dot((hypothesis(x,theta)-y).T,x).T
        
        # new cost
        cost1 = -(1/m)*cost(x,theta,y) # update cost function for logistic regression
        
        if it%100==0:
            print('Epoch:'+str(it)+' Cost:'+str(round(float(cost1),4)))
        cost_values.append(round(float(cost1),4))
        

    return theta,cost_values

def preprocess_df(df):
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


df = pd.read_csv('dataset_titanic.csv')
print('\t*** Dataset Info. ***')
print(df.info())
print('\n\n')

x,y = preprocess_df(df)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100)
print('** train ,test dataset **')
print('X_train shape: ',x_train.shape)
print('Y_train shape: ',y_train.shape)
print('\nX_test shape: ',x_test.shape)
print('Y_test shape: ',y_test.shape)


a=0.000069
epoch = 5000
theta_val,cost_val = gradient_descent(x_train,y_train,a, epoch)

# adding the extra col of 1s for b/x0 or theta0 like we did while training
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)
y_pred = hypothesis(x_test,theta_val)
y_pred = [1 if i>0.5 else 0 for i in y_pred] # setting a threshold for classification


y_test = np.reshape(y_test.values,(y_test.shape[0],1))
# print(y_test.shape,y_pred.shape)

print('\n\n\t*** Classification summary for the model on the test data. ***')

# calculate confusion matrix , and follow up error/
conf_matrix = confusion_matrix(y_test,y_pred)
print('\n** Confusion matrix **\n',conf_matrix)


# classification report:
report = classification_report(y_test,y_pred)
print('\n** Classification report **\n',report)

# plot the cost vs epoch graph
plt.plot(cost_val)
plt.show()
