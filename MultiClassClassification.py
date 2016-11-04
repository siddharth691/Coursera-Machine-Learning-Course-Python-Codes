from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import random

#Function Definitions
def lrcostFunctionReg(theta,reg,X,y):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(X))
    h = sigmoid(X.dot(theta))
    J = (-1)*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2*m))*sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        J[0] = np.inf
    return J[0]

def lrgradientReg(theta,reg,X,y):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(X))
    h = sigmoid(X.dot(theta))
    grad = (1/m)*X.T.dot(h-y)+(reg/m)*theta
    grad[0] = (1/m)*X[:,0].T.dot(h-y)
    return (grad.flatten())

def oneVsAll(features, classes, reg, n_labels):
    in_theta = np.zeros(features.shape[1])
    all_theta = np.zeros((n_labels,features.shape[1]))
    for c in range(1,n_labels+1):
        res = minimize(lrcostFunctionReg, in_theta, args=(reg,features,(classes==c)*1), method=None, jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1,:]=res.x

    return all_theta

def sigmoid(z):
    P = (1 / (1 + np.exp(-z)))
    return P

def predictOneVsAll(OptTheta, features):
    h = sigmoid(features.dot(OptTheta.T))
    y = np.zeros((features.shape[0],1))
    y = np.argmax(h,axis=1)
    y = y+1
    return y

def predictNN(theta1,theta2,features):
    z2 = theta1.dot(features.T)
    a2 = np.c_[np.ones((features.shape[0],1)),sigmoid(z2).T]
    z3 = theta2.dot(a2.T)
    a3 = sigmoid(z3)
    return (np.argmax(a3.T,axis =1)+1)

# Data Creation
data = loadmat('ex3data1.mat')
y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)),data['X']]

#Visualize the data
sample = np.random.choice(X.shape[0], 20)
SX= X[sample,1:]
plt.figure(1)
plt.imshow(SX.reshape(-1,20).T)
print y[sample,0]

#Fit
theta = oneVsAll(X,y,0.1,10)
#Prediction and Accuracy
pred = predictOneVsAll(theta, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

# Predict and accuracy for Neural Networks
weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

pred = predictNN(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))
plt.show()

