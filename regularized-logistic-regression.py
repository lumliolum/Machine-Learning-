# implementation of regularized logistic regression

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.linear_model import LogisticRegression
# loading data
data=np.loadtxt("data-regularized-logistic.txt",delimiter=',')
X=data[:,0:2];
y=data[:,2:3];
(m,n)=X.shape;

# setting the values of the constants
l=0.01;                 #lambda
alpha=0.003
iterations=500000


y.shape=(len(y),)
# ploting data
def plotdata(X,y):
    (m,n)=X.shape
    plt.axis([-1,1.5,-0.8,1.2])
    plt.title("data visualization")
    plt.xlabel("Mircochip test1")
    plt.ylabel("Mircochip test2")
    pos1=(X[y==1,0])
    pos2=(X[y==1,1])
    neg1=(X[y==0,0])
    neg2=(X[y==0,1])
    plt.plot(pos1,pos2,'r+',label='y=1')
    plt.plot(neg1,neg2,'bo',label='y=0')
    plt.legend()
    plt.show()

# feature mapping
def mapfeature(X,degree):
    (m,n)=X.shape
    x1=X[:,0:1]
    x2=X[:,1:2]
    K=np.hstack((x1,x2))
    print(K.shape)
    i=2;
    while(i<=degree):
        j=0
        while(j<=i):
            K=np.hstack(( K,(x1**(i-j))*( x2**(j)) ))
            j=j+1
        i=i+1
    # adding bias term
    K=np.hstack((np.ones((m,1)),K))
    return K;


def mapfeature1(X,degree):
    k=np.zeros((28,1))
    X.shape=(1,2)
    x1=X[0,0]
    x2=X[0,1]
    k[0]=1
    k[1]=x1
    k[2]=x2
    i=2
    c=3
    while(i<=degree):
        j=0
        while(j<=i):
            k[c]=(x1**(i-j))*(x2**j)
            c=c+1
            j=j+1
        i=i+1
    return k


def plotCost(J,iterations):
    x=np.arange(iterations+1)
    plt.plot(x,J)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost Function")
    plt.show()

def hypo(X,theta):
    h=1/(1+np.exp(-(X.dot(theta))))               #sigmoid function
    return h;

# regularized costfunction
def costfunction(theta,X,y,l):
    (m,n)=X.shape
    J=-(y*np.log(hypo(X,theta))+(1-y)*(np.log(1-hypo(X,theta))))
    J=(np.sum(J))/(m)
    theta_squared=theta*theta
    theta_squared.shape=(n,)
    J=J+(l/(2*m))*(np.sum(theta_squared)-theta_squared[0])
    return J;


def gradient(theta,X,y,l):
    (m,n)=X.shape
    k=hypo(X,theta)
    x=l/m
    sum_added=x*theta
    grad=(1/m)*np.dot((X.transpose()),(k-y)) + sum_added
    grad[0]=grad[0]-(l/m)*theta[0]
    return grad



def gradientdescent(theta,X,y,l,alpha,iterations):
    (m,n)=X.shape
    J_val=np.linspace(0,0,iterations+1)
    for i in range(iterations+1):
        J_val[i]=costfunction(theta,X,y,l)
        theta=theta-(alpha)*gradient(theta,X,y,l)
    plotCost(J_val,iterations)
    print(np.min(J_val))
    return theta;


# plotting the result
def plotresult(X,y,theta,X_map_feature):
    (m,n)=X_map_feature.shape
    pos1=(X[y==1,0])
    pos2=(X[y==1,1])
    neg1=(X[y==0,0])
    neg2=(X[y==0,1])
    plt.axis([-1,1.5,-0.8,1.2])
    plt.title("Result with decision boundary")
    plt.xlabel("Microchip Test1")
    plt.ylabel("Microchip Test2")
    plt.plot(pos1,pos2,'r+',label='y=1',)
    plt.plot(neg1,neg2,'bo',label='y=0')
    # we have to plot the decision boundary
    x1list=np.linspace(-1,1.5,50)
    X,Y=np.meshgrid(x1list,x1list)
    a=len(x1list)
    z=np.zeros((a,a))
    x1list.shape=(a,1)
    for i in range(a):
        u=x1list[i,0]
        for j in range(a):
            v=x1list[j,0]
            arr=np.hstack((u,v))
            arr_map=mapfeature1(arr,6)
            m=np.dot(theta.reshape(1,n),arr_map)
            z[i,j]=m[0,0]
    plt.contour(X,Y,z,[0])
    plt.legend()
    plt.show()

#plotting the data
plotdata(X,y)

# feature mapping
X_old=X
X=mapfeature(X,6)               # will be of 118*28
(m,n)=X.shape                 # m=118 and n=28 with bias sterm included
initial_theta=np.zeros((n,))

# print(costfunction(initial_theta,X,y,l))     # expected cost 0.693147
# print(gradient(initial_theta,X,y,l))

theta=gradientdescent(initial_theta,X,y,l,alpha,iterations)
print(theta.shape)
plotresult(X_old,y,theta,X)



# using scikit learn
# model = LogisticRegression()
#
# model.fit(X,y)
# theta2=model.coef_.reshape((n,))
# print(costfunction(theta2,X,y,l))
