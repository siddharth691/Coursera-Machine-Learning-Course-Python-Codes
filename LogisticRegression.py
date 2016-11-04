import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def costFunction(theta,X,y):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(X))
    h = sigmoid(X.dot(theta))
    J = (-1)*(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        J[0] = np.inf
    return J[0]

def gradient(theta,X,y):
    theta = np.array(theta,dtype ='float')
    theta = theta.reshape(-1,1)
    m = float(len(X))
    h = sigmoid(X.dot(theta))
    grad = (1/m)*X.T.dot(h-y)
    return (grad.flatten())

def plotDecisionBoundary(OptTheta,data,fig,label_x,label_y,pos_label,neg_label,dec_label):
    OptTheta = np.array(OptTheta,dtype='float')
    neg = data[data[:,2]==0]
    pos = data[~(data[:,2]==0)]
    plt.figure(fig)
    plt.scatter(pos[:,0],pos[:,1],marker='+', c='k', s=60, linewidth=2, label=pos_label)
    plt.scatter(neg[:,0],neg[:,1],c='y', s=60, label=neg_label)
    dec_x = np.array([min(X[:,1])-2,max(X[:,1])+2])
    dec_y = (-1)*((OptTheta[0]/OptTheta[2])+(OptTheta[1]/OptTheta[2])*dec_x)
    plt.plot(dec_x,dec_y,c = 'b',linewidth=3,label=dec_label)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()

def predict(X,OptTheta):
    OptTheta= np.array(OptTheta)
    OptTheta= OptTheta.reshape(-1,1)
    p = np.empty(len(X))
    h = sigmoid(X.dot(OptTheta))
    p[h[:,0]>=0.5] = 1
    p[h[:,0]<0.5]=0
    return p

#Data Creation
data = np.loadtxt('ex2data1.txt', delimiter = ',')
X = np.c_[np.ones((data.shape[0],1)),data[:,0],data[:,1]]
y = np.c_[data[:,2]]

#Visualize Data
plotdata(data,'Exam 1 Score','Exam 2 Score','Admitted','Not Admitted',1)

#Initial Cost and Gradient Values
initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print 'Initial Cost and gradient values'
print 'Cost: \n', cost
print 'Grad: \n', grad

#Fit
res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
print res
#Plot Decision Boundary
plotDecisionBoundary(res.x,data,2,'Exam 1 Score','Exam 2 Score','Admitted','Not Admitted','Decision Boundary')
#Predict
prob = sigmoid(res.x*[1, 45, 85])
print 'For a student with scores 45 and 85, we predict an admission probability of ',prob
#Training Accuracy
p = predict(X,res.x)
true = [p[i] for i in range(len(X)) if p[i]==y[i]]
Taccuracy = (len(true))/float(len(X))
print 'training Accuracy = ', Taccuracy*100

plt.show()

    
    
    



    
