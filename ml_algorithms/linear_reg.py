
'''
Vectorized implementation of gradient descent algorithm.
Important points : 
        - fourmula of cost fucntion , partial derivatives
        - make sure the vector dimensions are as expected before doing numpy operations 

By - 0x1h0b

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def gradient_descent(x,y, alpha, epoch):
    m,n = x.shape[0],x.shape[1] # no of training examples, no of features
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    theta = np.ones((n+1,1))
    n = n+1 # +1 cause we added the theta0 or corresponding x0 (intercept) to x
    cost_values=[]
    
    y = np.reshape(y.values,(m,1)) # previous shape (m,)
    # starting algo
    print('\tstarting gradient descent!')
    for it in range(epoch):

        # while calulating theta , trying to reduce a for loop by using vectors
        theta -= alpha*(1/m)*np.dot((np.dot(x,theta)-y).T,x).T
        
        # new cost
        cost = (1/2*m)*np.sum(np.square(np.dot(x,theta)-y))
        if it%100==0:
            print('Epoch:'+str(it)+' Cost:'+str(round(cost,3)))
        cost_values.append(cost)
        

    return theta,cost_values


df = pd.read_csv('dataset_reg.csv')
print('\tDataset Info.')
print(df.info())
print('\n\n')

x = df.drop(' Y',axis=1)
y=df[' Y']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100)
print('X_train shape: ',x_train.shape)
print('Y_train shape: ',y_train.shape)
print('\nX_test shape: ',x_test.shape)
print('Y_test shape: ',y_test.shape)


a=0.09
epoch = 5000
theta_val,cost_val = gradient_descent(x_train,y_train,a, epoch)

# adding the extra col of 1s for b/x0 or theta0 like we did while training
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)
y_pred = np.dot(x_test,theta_val)

y_test = np.reshape(y_test.values,(y_test.shape[0],1))
# print(y_test.shape,y_pred.shape)
score = r2_score(y_test,y_pred)
print("r2 score: ",score)

# plot the cost vs epoch graph

plt.plot(cost_val)
plt.show()
