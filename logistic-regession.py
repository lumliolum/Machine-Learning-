# implementation of logistic regression.
# the simplest binary classfication
# It is non-reguralized implementation.
# classfication  is based on only two features.

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
# data load
data=np.loadtxt('data-non-regularized-logistic.txt',delimiter=',',usecols=range(3))
X=data[:,0:2]           #first two columns
y=data[:,2:3]           # last columns and labels.
(m,n)=X.shape

# setting up the values
alpha=0.001          #0.003
iterations=1000000        # 100,000

#function which will plot the data
def plotdata(X,y):
    plt.axis([30,100,30,100])
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.title("Data visualization")
    (m,n)=X.shape
    k=y.reshape(m,)
    pos1=(X[k==1,0])
    pos2=(X[k==1,1])
    neg1=(X[k==0,0])
    neg2=(X[k==0,1])
    plt.plot(pos1,pos2,'r+',label='admiited')
    plt.plot(neg1,neg2,'bo',label='not admitted')
    plt.legend()
    plt.show()

def plotCost(J,iterations):
    x=np.arange(iterations+1)
    plt.plot(x,J)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost Function")
    plt.show()

def plotresult(X,y,theta):
    plt.axis([30,100,30,100])
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.title("result")
    (m,n)=X.shape
    k=y.reshape(m,)
    pos1=(X[k==1,0])
    pos2=(X[k==1,1])
    neg1=(X[k==0,0])
    neg2=(X[k==0,1])
    plt.plot(pos1,pos2,'r+',label='admiited')
    plt.plot(neg1,neg2,'bo',label='not admitted')
    x1_data=np.linspace(30,100,2000)
    plt.plot(x1_data,-(theta[0,0]+theta[1,0]*x1_data)/theta[2,0],label='Decision Boundary')
    plt.legend()
    plt.show()

# hypothesis function
def hypo(X,theta):
    h=1/(1+np.exp(-(X.dot(theta))))               #sigmoid function
    return h;

def costfunction(theta,X,y):
    (m,n)=X.shape
    J=-(y*np.log(hypo(X,theta))+(1-y)*(np.log(1-hypo(X,theta))))
    J=np.sum(J)
    J=J/m
    return J;

def gradientDescent(X,y,theta,alpha,iterations):
    (m,n)=X.shape
    J_val=np.linspace(0,0,iterations+1)
    for i in range(iterations+1):
        J_val[i]=costfunction(theta,X,y)
        theta=theta-(alpha/m)*np.dot((X.transpose()),(hypo(X,theta)-y))
    plotCost(J_val,iterations)
    return theta;

def gradient(theta,X,y):
    (m,n)=X.shape
    k=hypo(X,theta).reshape((m,1))
    grad=np.dot((X.transpose()),(k-y))
    grad=grad/m
    return grad.flatten();


# visualizing the data
plotdata(X,y)


X_old=X
add=np.ones((m,1))
X=np.hstack((add,X))           # adding the bias term to the feature matrix.


initial_theta=np.zeros((n+1,1))        # one is for bias term and theta is a column vector

print(X.shape,y.shape,initial_theta.shape)

print(costfunction(initial_theta,X,y))    # just checking that cost function is working fine.
theta1=gradientDescent(X,y,initial_theta,alpha,iterations)
print(theta1)
# will plot the decision boundary
plotresult(X_old,y,theta1)

# the function is minimized using minimize
k=y.reshape(m,)
# theta2=op.minimize(fun=costfunction,x0=initial_theta,args=(X,k),method='TNC')
# theta2=theta2.x
# print(theta2);
# theta2=np.array(theta2).reshape((len(theta2),1))
# plotresult(X_old,y,theta2)
