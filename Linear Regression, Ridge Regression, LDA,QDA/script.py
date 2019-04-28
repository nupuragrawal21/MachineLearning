import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    y=y.reshape(len(y))
    mean_matrix=np.zeros((2,5)) 
    for row in range(0,mean_matrix.shape[0]):
        for col in range(0,mean_matrix.shape[1]):
            #print(row,col)
            mean_matrix[row,col]=X[y==col+1,row].mean()
    # covmat - A single d x d learnt covariance matrix 
    covmat=np.cov(X[:,0],X[:,1])
    return mean_matrix,covmat

def qdaLearn(X,y):
    y=y.reshape(len(y))
    mean_matrix=np.zeros((2,5)) 
    for row in range(0,mean_matrix.shape[0]):
        for col in range(0,mean_matrix.shape[1]):
            #print(row,col)
            mean_matrix[row,col]=X[y==col+1,row].mean()
    covmats=[]
    det_covmats=np.zeros(mean_matrix.shape[1])
    for i in range(0,len(np.unique(y))):
                    #print(i)
                    cov=np.cov(X[y==i+1,0],X[y==i+1,1])
                    covmats.append(cov)
                    det_covmats[i]=np.linalg.det(cov)
    #print(type(mean_matrix))
    return mean_matrix,covmats,det_covmats

def ldaTest(means,covmat,Xtest,ytest):
    mah=[0,0,0,0,0]
    
    for i in range(0,means.shape[1]):
            m_1=np.matmul((Xtest-means[:,i]),np.linalg.inv(covmat))
            maha=np.matmul(m_1,np.transpose(Xtest-means[:,i]))
            mah[i]=np.exp(-0.5*maha)
    y_class=[]
    
    for n in range(0,len(mah)):
        class_num=np.diag(mah[n])
        y_class.append(class_num)
    y_class= np.array(y_class)
    y_pred= np.zeros(y_class.shape[1])
    for row in range(0,y_class.shape[1]):
            #print(row)
            y_pred[row]=np.argmax(y_class[:,row]) 
    ypred=y_pred+1
    ytest=ytest.reshape(len(ytest))
    acc = (ytest==ypred).mean()
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    mah=[0,0,0,0,0]
    ypred= np.zeros((Xtest.shape[0]))
    y_class=[]
    for i in range(0,means.shape[1]):
            sub_term = Xtest-means[:,i]
            m_1=np.dot(sub_term,np.linalg.inv(covmats[i]))
            maha=np.dot(m_1,np.transpose(sub_term))
            denom = np.linalg.det(covmats[i])
            mah=np.exp(-0.5*maha.diagonal())
            class_num=mah/denom
            y_class.append(class_num)
    y_class= np.array(y_class)
    y_pred= np.zeros(y_class.shape[1])
    for row in range(0,y_class.shape[1]):
            #print(row)
            y_pred[row]=np.argmax(y_class[:,row]) 
    ypred=y_pred+1
    
    ytest=ytest.reshape(len(ytest))
    acc = (ytest==ypred).mean()
    return acc,ypred

def learnOLERegression(X,y):   
    X_trans = np.transpose(X) 
    result1 = np.matmul(X_trans, X)
    inverse = np.linalg.inv(result1)
    result2 = np.matmul(inverse, X_trans)
    w = np.matmul(result2, y)
    return(w)

def learnRidgeRegression(X,y,lambd):
    X_trans = np.transpose(X)
    term1 = np.matmul(X_trans,X)
    N = X.shape[1]
    I= np.identity(N)
    term2 = lambd*I
    result1 = term1 +term2
    #print("shape : ",result1.shape)
    inverse = np.linalg.inv(result1)
    result2= np.matmul(X_trans,y)
    w=np.matmul(inverse,result2)
    return w



def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    n=x.shape[0]
    M=np.ones((n, p+1))
    for i in range(1,p+1):
        M[:,i]=np.power(x, i)
    return(M)
    
# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
