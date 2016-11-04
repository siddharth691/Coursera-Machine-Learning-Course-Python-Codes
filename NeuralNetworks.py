from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

#Function Definitions
def sigmoid(z):
    P = (1 / (1 + np.exp(-z)))
    return P

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def nnCostFunction(params,SizeL1,SizeL2,SizeL3, reg, X, y):
    m = float(X.shape[0])
    theta1 = params[0:(SizeL2*(SizeL1+1))].reshape(SizeL2,(SizeL1+1))
    theta2 = params[(SizeL2*(SizeL1+1)):].reshape(SizeL3,(SizeL2+1))
    a1 = X.T
    z2  = theta1.dot(X.T)
    a2  = (np.c_[np.ones((X.shape[0],1)),sigmoid(z2).T]).T
    z3  = theta2.dot(a2)
    a3  = sigmoid(z3)
    h = a3
    yb = np.zeros((SizeL3,X.shape[0]))
    for i in range(X.shape[0]):
        yb[y[i]-1,i]=1
    J = (1/m)*np.sum(np.sum((-yb*np.log(h) - (1-yb)*np.log(1-h)),axis = 0))+(reg/(2*m))*(np.sum(np.square(theta1[:,1:]))+np.sum(np.square(theta2[:,1:])))

    d3 = a3 - yb
    d2 = theta2[:,1:].T.dot(d3)*sigmoidGradient(z2)

    delta2 = d3.dot(a2.T)
    delta1 = d2.dot(a1.T)

    theta1_ = np.c_[np.zeros((theta1.shape[0],1)),theta1[:,1:]]
    theta2_ = np.c_[np.zeros((theta2.shape[0],1)),theta2[:,1:]]

    theta1_grad = delta1/m + (theta1_*reg)/m
    theta2_grad = delta2/m + (theta2_*reg)/m
    grad = np.empty(params.size,dtype='int8')
    grad = np.r_[theta1_grad.ravel(),theta2_grad.ravel()]

    return(J, grad)

def computeNumericalGradient(costFunction, theta):
    e = 0.0001
    for i in range(len(theta)):
        thetaplus = theta
        thetaplus[i] = thetaplus[i] + e
        thetaminus = theta
        thetaminus[i] = thetaminus[i] - e
        Ngrad[i] = (costFunction(thetaplus) - costFunction(thetaminus))/2*e

    return Ngrad
            


def randInitializeWeights(sizeLin, sizeLout):
    ep = 0.12
    theta = np.random.random((sizeLout,sizeLin+1))*(2*ep) - ep
    return theta

def predictNN(params,SizeL1,SizeL2,SizeL3,features):
    theta1 = params[0:(SizeL2*(SizeL1+1))].reshape(SizeL2,(SizeL1+1))
    theta2 = params[(SizeL2*(SizeL1+1)):].reshape(SizeL3,(SizeL2+1))
    z2 = theta1.dot(features.T)
    a2 = np.c_[np.ones((features.shape[0],1)),sigmoid(z2).T]
    z3 = theta2.dot(a2.T)
    a3 = sigmoid(z3)
    return (np.argmax(a3.T,axis =1)+1)


#Data Creation
data  = loadmat('ex4data1')
y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)),data['X']]

weights = loadmat('ex4weights')
theta1,theta2 = weights['Theta1'], weights['Theta2']
params = np.r_[theta1.ravel(),theta2.ravel()]
  
#Cost Function with loaded theta values
print 'Reg Cost Function value for given thetas: ',nnCostFunction(params,400,25,10,1,X,y)

#Fit
SizeL1 = 400
SizeL2 = 25
SizeL3 = 10
reg = 1
Theta1 = randInitializeWeights(SizeL1,SizeL2)
Theta2 = randInitializeWeights(SizeL2,SizeL3)
in_theta = np.r_[Theta1.ravel(),Theta2.ravel()]
res = minimize(nnCostFunction, in_theta, args=(SizeL1,SizeL2,SizeL3, reg, X, y), method=None, jac=True, options={'maxiter':50})

#Predict and Accuracy
pred = predictNN(res.x,SizeL1,SizeL2,SizeL3,X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))
