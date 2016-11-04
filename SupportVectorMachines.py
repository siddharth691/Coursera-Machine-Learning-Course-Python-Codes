import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Function Definitions
def plotData(X,y,figNo):
    pos = X[(y==1).reshape(-1,),:]
    neg = X[(y==0).reshape(-1,),:]    
    plt.figure(figNo)
    plt.scatter(pos[:,0],pos[:,1],c='k',s=60,linewidths=1,marker='+')
    plt.scatter(neg[:,0],neg[:,1],c='y',marker='o',linewidths=1,s=60)

def plot_svc(clf,X,y,figNo,pad =0.25,h=0.02):
    xmin,xmax = X[:,0].min()-pad,X[:,0].max()+pad
    ymin,ymax = X[:,1].min()-pad,X[:,1].max()+pad
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
    plt.figure(figNo)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plotData(X,y,figNo)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha = 0.5)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    
def OptimalParams(X,y,Xval,yval):
    Csteps = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    SigSteps = 1/np.square(Csteps)
    OptC= np.inf
    OptSigma = np.inf
    maxAcc = 0
    for CurC in Csteps:
        for CurSigma in SigSteps:
            clf = SVC(C = CurC, kernel = 'rbf', gamma = CurSigma)
            clf.fit(X,y.ravel())
            pred = clf.predict(Xval)
            acc = accuracy_score(yval,pred)
            if acc> maxAcc:
                maxAcc = acc
                OptC = CurC
                OptSigma = CurSigma
    return (OptC,OptSigma)
        
#Data Creation
data1 = loadmat('ex6data1')
X1 = data1['X']
y1 = data1['y']

#Visualize 2 Dimension Data
plotData(X1,y1,1)

#Fit using svm classifier using Linear Kernel for different C and Plot
clf1 = SVC(C=1.00,kernel='linear')
clf1.fit(X1,y1.ravel())
plot_svc(clf1,X1,y1,2)# Just Right Ignoring the outlier


clf2 = SVC(C=100,kernel='linear')
clf2.fit(X1,y1.ravel())
plot_svc(clf2,X1,y1,3)#OverFit reg very small, high Variance

#Load DataSet 2
data2 = loadmat('ex6data2')
X2  = data2['X']
y2 = data2['y']

#Visualise dataset 2
plotData(X2,y2,4)

#Fit Using Gaussian Kernel 'rbf'
clf3 = SVC(C = 50, gamma = 6 ,kernel='rbf')
clf3.fit(X2,y2.ravel())
plot_svc(clf3,X2,y2,4)

#Load Dataset 3 and Visualise
data3 = loadmat('ex6data3')
X3 = data3['X']
y3 = data3['y']
Xval = data3['Xval']
yval = data3['yval']
plotData(X3,y3,5)

# Find optimal parameters of C and gamma using Cross Validation Set
OptC,OptSigma = OptimalParams(X3,y3,Xval,yval)
clf4 = SVC(C = OptC, kernel ='rbf',gamma = OptSigma)
clf4.fit(X3,y3.ravel())
pred = clf4.predict(Xval)
plot_svc(clf4,X3,y3,6)
print 'Accuracy of the classifier with optimal parameters is :',accuracy_score(yval,pred)
print 'Optimal C used is :',OptC
print 'Optimal gamma Used is :',OptSigma

plt.show()
