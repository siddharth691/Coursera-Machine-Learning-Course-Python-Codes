import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

#Function Definitions
def plotdata(data,label_x,label_y,pos_label,neg_label,fig):
    neg = data[data[:,2]==0]
    pos = data[~(data[:,2]==0)]
    plt.figure(fig)
    plt.scatter(pos[:,0],pos[:,1],marker='+', c='k', s=60, linewidth=2, label=pos_label)
    plt.scatter(neg[:,0],neg[:,1],c='y', s=60, label=neg_label)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()

def sigmoid(z):
    P = (1 / (1 + np.exp(-z)))
    return P

def costFunctionReg(theta,reg,*arg):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(XX))
    h = sigmoid(XX.dot(theta))
    J = (-1)*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y)) + (reg/(2*m))*sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        J[0] = np.inf
    return J[0]

def gradientReg(theta,reg,*arg):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(XX))
    h = sigmoid(XX.dot(theta))
    grad = (1/m)*XX.T.dot(h-y)+(reg/m)*theta
    grad[0] = (1/m)*XX[:,0].T.dot(h-y)
    return (grad.flatten())

def plotDecisionBoundary(OptTheta,XX,y,axes,label_x,label_y,pos_label,neg_label,dec_label):
    OptTheta = np.array(OptTheta,dtype='float')
    neg = XX[y[:,0]==0]
    pos = XX[~(y[:,0]==0)]
    axes.scatter(pos[:,1],pos[:,2],marker='+', c='k', s=60, linewidth=2, label=pos_label)
    axes.scatter(neg[:,1],neg[:,2],c='y', s=60, label=neg_label)
    x1 = np.linspace(min(XX[:,1]),max(XX[:,1]))
    x2 = np.linspace(min(XX[:,2]),max(XX[:,2]))
    xx1, xx2 = np.meshgrid(x1,x2)
    X = poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()])
    h = sigmoid(X.dot(OptTheta))
    axes.contour(xx1,xx2,h.reshape(xx1.shape),[0.5], linewidths=1, colors='g')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()

def predict(XX,OptTheta):
    OptTheta= np.array(OptTheta)
    OptTheta= OptTheta.reshape(-1,1)
    p = np.empty(len(XX))
    h = sigmoid(XX.dot(OptTheta))
    p[h[:,0]>=0.5] = 1
    p[h[:,0]<0.5]=0
    return p


#Data Creation
data = np.loadtxt('ex2data2.txt', delimiter = ',')
X = np.c_[data[:,0],data[:,1]]# ones not concatenated because polynomial features to be added
poly = PolynomialFeatures(6)
XX = poly.fit_transform(X) # 6 degree polynomial features taking total features to 28
y = np.c_[data[:,2]]

#Visualize Data
plotdata(data,'Microchip Test 1','Microchip Test 2','y=1','y=0',1)

#Initial Cost Function and Gradient Check
initial_theta = np.zeros(XX.shape[1])
print costFunctionReg(initial_theta, 1, XX, y)
print gradientReg(initial_theta, 1, XX, y)

#Simultaneous Fit and Decision Boundary Plot
fig = plt.figure(2)
ax = []
for i, C in enumerate([0, 1, 100]):
    print 'XX.shape = ',XX.shape
    print 'y.shape = ',y.shape
    print 'initial_theta.shape =',initial_theta.shape
    res = minimize(costFunctionReg, initial_theta, args=(C,XX,y), method=None, jac=gradientReg, options={'maxiter':3000})
    axCount = 131
    ax = fig.add_subplot(axCount+i)
    plotDecisionBoundary(res.x,XX,y,ax,'Microchip Test 1','Microchip Test 2','y = 1','y = 0','Decision Boundary')
    p = predict(XX,res.x)
    true = [p[i] for i in range(len(XX)) if p[i]==y[i]]
    Taccuracy = (len(true))/float(len(XX))
    ax.set_title('Accuracy with reg={}: {}%'.format(C,np.round(Taccuracy*100,decimals=2)))
        
plt.show()   
