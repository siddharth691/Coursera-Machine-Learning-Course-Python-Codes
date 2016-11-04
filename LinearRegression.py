import numpy as np
import matplotlib.pyplot as plt
#Data Creation
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

#Writing Functions
def computeCost(X,y,theta = [[0],[0]]):
    theta = np.array(theta, dtype ='float')
    m = float(len(X))
    theta = theta.reshape(1,2)
    h = np.sum(theta*X,axis=1)
    J = 0
    J = (1/(2*m))*np.sum(np.square(h.reshape(len(X),1)-y))
    return J

def gradientDescent(X,y,theta=[[0],[0]],alpha=0.01,iterations=1500):
    theta = np.array(theta,dtype = 'float')
    theta = theta.reshape(1,2)
    m = float(len(X))
    J_history = np.zeros(iterations)
    for i in range(iterations):
        h = np.sum(theta*X,axis=1)
        h = h.reshape(len(X),1)
        grad = np.array(np.zeros(theta.size))
        for j in range(theta.size):
            grad[j] = (1/m)*(sum((h-y)*(X[:,j].reshape(len(X),1)))) 
            theta[0,j] += - (alpha*(grad[j]))
        J_history[i] = computeCost(X, y, theta)
    return (theta, J_history)
#Fit
#theta is numpy array of shape (1,2)
theta , Cost_J = gradientDescent(X, y)
#Plot
plt.figure(1)
plt.plot(range(1500),Cost_J)
plt.figure(2)
h = np.sum(theta*X,axis=1)
h = h.reshape(len(X),1)
plt.scatter(X[:,1], y, c='r', marker='x',label='Training Data')
plt.plot(X[:,1],h,c='b',label='Linear Regression')
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.legend()
#Predict
f1 = np.c_[1,3.5]
f2 = np.c_[1,7]
print 'Profit for city with population of 35000: ',np.sum(theta*f1*10000)
print 'Profit for city with population of 70000: ',np.sum(theta*f2*10000)

plt.show()
        
        
    
