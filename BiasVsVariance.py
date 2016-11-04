from scipy.io import loadmat
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Function  Definitions
def linearRegCostFunction(theta,X,y,reg):
    theta = np.array(theta).reshape(-1,1)
    m = float(X.shape[0])
    J = (1/(2*m))*(np.sum(np.square(X.dot(theta) - y)) +reg*(np.sum(np.square(theta[1:]))))
    return J
def linearRegGradient(theta,X,y,reg):
    theta = np.array(theta).reshape(-1,1)
    m = float(X.shape[0])
    grad = (1/m)*(np.sum((X.dot(theta) - y)*X, axis = 0)).reshape(1,-1) + (reg/m)*(np.c_[0,theta[1:,0].reshape(1,-1)])
    return grad.flatten()

def trainLinearReg(X,y,reg):
    in_theta = np.zeros(X.shape[1])
    res = minimize(linearRegCostFunction, in_theta, args=(X, y,reg), method=None, jac=linearRegGradient, options={'maxiter':200})
    return (res.x).reshape(-1,1)

def learningCurve(X,y,Xval,yval,reg):
    m = X.shape[0]
    error_train = []
    error_val = []
    for i in range(1,m+1):
        theta = trainLinearReg(X[:i,:],y[:i,:],reg)
        error_train.append(linearRegCostFunction(theta,X[:i,:],y[:i,:],0))
        error_val.append(linearRegCostFunction(theta,Xval,yval,0))
    return (np.array(error_train), np.array(error_val))

def polyfeatures(X,p):
    Xpoly = np.empty((X.shape[0],p))
    powers = np.matlib.repmat(np.array(range(1,p+1)),X.shape[0],1)
    Xrepeated = np.matlib.repmat(X,1,p)
    Xpoly = np.power(Xrepeated,powers)
    return Xpoly

def degreeCurve(X,y,Xval,yval,p,reg):
    error_train = []
    error_val = []
    for i in range(1,p+1):
        Xpoly = polyfeatures(X,i)
        mu = np.mean(Xpoly,axis = 0).reshape(1,-1)
        sigma = np.std(Xpoly,axis =0)
        sigma = sigma.astype(float)
        Xpoly = (Xpoly - mu)/sigma
        theta = trainLinearReg(np.c_[np.ones((X.shape[0],1)),Xpoly],y,reg)
        error_train.append(linearRegCostFunction(theta,np.c_[np.ones((X.shape[0],1)),Xpoly],y,0))
        XvalPoly = polyfeatures(Xval,i)
        XvalPoly = (XvalPoly - mu)/sigma
        error_val.append(linearRegCostFunction(theta,np.c_[np.ones((Xval.shape[0],1)),XvalPoly],yval,0))
    return (np.array(error_train), np.array(error_val))

def lamdaCurve(X,y,Xval,yval,deg,lamda):
    error_train=[]
    error_val =[]
    Xpoly = polyfeatures(X,deg)
    mu = np.mean(Xpoly,axis = 0).reshape(1,-1)
    sigma = np.std(Xpoly,axis =0)
    sigma = sigma.astype(float)
    Xpoly = (Xpoly - mu)/sigma
    XvalPoly = polyfeatures(Xval,deg)
    XvalPoly = (XvalPoly - mu)/sigma
    for i in lamda:
        theta = trainLinearReg(np.c_[np.ones((X.shape[0],1)),Xpoly],y,i)
        error_train.append(linearRegCostFunction(theta,np.c_[np.ones((X.shape[0],1)),Xpoly],y,0))
        error_val.append(linearRegCostFunction(theta,np.c_[np.ones((Xval.shape[0],1)),XvalPoly],yval,0))
    return (np.array(error_train), np.array(error_val))
    
#Data Creation
data = loadmat('ex5data1')
data.keys()
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']

#Visualize Data
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax1.scatter(X,y,s=20,marker='x',c='r')
ax1.set_xlabel('Change in water level(x)')
ax1.set_ylabel('Water flowing out of the dam(y)')
ax1.set_xlim((-50,40))
ax1.set_ylim((0,40))

#First Quick fit Linear Regression and plot(w/ only 1 feature and reg = 0)
reg = 0
OptTheta = trainLinearReg(np.c_[np.ones((X.shape[0],1)),X],y,reg)
ax2 = fig.add_subplot(212)
ax2.scatter(X,y,s=20,marker='x',c='r',label='Training Data')
ax2.set_xlabel('Change in water level(x)')
ax2.set_ylabel('Water flowing out of the dam(y)')
ax2.plot(X,np.c_[np.ones((X.shape[0],1)),X].dot(OptTheta),c='b',label='Linear Regression')
ax2.set_xlim((-50,40))
ax2.set_ylim((-5,40))

#Plot Learning Curve to check Bias/Variance Problem
error_train, error_val = learningCurve(np.c_[np.ones((X.shape[0],1)),X],y,np.c_[np.ones((Xval.shape[0],1)),Xval],yval,reg)
plt.figure(2)
plt.plot(range(1,X.shape[0]+1),error_train.ravel(),c = 'b', label = 'Training Error')
plt.plot(range(1,X.shape[0]+1),error_val.ravel(),c='r',label='Cross Validation Error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()# Graph confirms the problem is a bias problem, so we can add more features to correct this problem

# Model Selection Plot Degree vs Error Curve
# Selection of optimal degree with zero reguralization parameter
p = 8
error_train, error_val = degreeCurve(X,y,Xval,yval,p,0)
plt.figure(3)
plt.plot(range(1,p+1),error_train.ravel(),c='b',label='Training Error')
plt.plot(range(1,p+1),error_val.ravel(),c='r',label='Cross Validation Error')
plt.xlabel('Degree')
plt.ylabel('Error')
plt.legend()# This curve shows minimum Cross validation error is at degree = 3, So we find reg on deg = 3 param

#Model Selection Plot Reg vs Error Curve
#Selection of optimal reg for degree  = 3

deg = 3
lamda = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
error_train, error_val = lamdaCurve(X,y,Xval,yval,deg,lamda)
plt.figure(4)
plt.plot(lamda,error_train.ravel(),c='b',label='Training Error')
plt.plot(lamda,error_val.ravel(),c='r',label='Cross Validation Error')
plt.xlabel('Reguralization Parameter')
plt.ylabel('Error')
plt.legend()#It shows reg is minimum lamda is 1

#Computing Test Set Error
OptDeg = 3
OptReg = 1
Xpoly = polyfeatures(X,OptDeg)
mu = np.mean(Xpoly,axis = 0).reshape(1,-1)
sigma = np.std(Xpoly,axis =0)
sigma = sigma.astype(float)
Xpoly = (Xpoly - mu)/sigma
XvalPoly = polyfeatures(Xval,OptDeg)
XvalPoly = (XvalPoly - mu)/sigma
XtestPoly = polyfeatures(Xtest,OptDeg)
XtestPoly = (XtestPoly - mu)/sigma
Opttheta = trainLinearReg(np.c_[np.ones((Xpoly.shape[0],1)),Xpoly],y,OptReg)
TestError= linearRegCostFunction(Opttheta,np.c_[np.ones((XtestPoly.shape[0],1)),XtestPoly],ytest,0)
print 'Final Test Error after model selection is = ',TestError

#Final Visualization
plt.figure(5)
plt.scatter(X,y,s=20,marker='x',c='r',label='Training Data')
plt.xlabel('Change in water level(x)')
plt.ylabel('Water flowing out of the dam(y)')
Xsort = np.sort(X,axis=0)
XpolySort = polyfeatures(Xsort,OptDeg)
XpolySort = (XpolySort - mu)/sigma
plt.plot(Xsort,np.c_[np.ones((XpolySort.shape[0],1)),XpolySort].dot(Opttheta),c='b',label='Linear Regression')


plt.show()
