# program for fitting the straight line for the data
# implementation of linear regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


# for loading the data
x=np.loadtxt('data.txt',delimiter=',',usecols=range(2))

X_old=x[:,0:1]           #first column
y=x[:,1:2]              #second column
(m,n)=x.shape
add=np.ones((m,1))

#setting up the values
alpha=0.01
iterations=1500

#function for plot
def plotdata(X,y):
    plt.axis([4,25,-5,25])
    plt.plot(X.reshape(1,m),y.reshape(1,m),'rx')
    plt.xlabel("Population of city in 10,000's")
    plt.ylabel("Profit in $10,000")
    plt.title("Data visualization")
    plt.show()
    return;

#will plot the value of costfuncction as a function of iterations
def plotCost(J,iterations):
    x=np.arange(iterations+1)                 # will create a array from 0 to iterations..
    plt.plot(x,J)
    plt.axis([0,10,0,40])
    plt.xlabel("number of iterations")
    plt.ylabel("cost Function")
    plt.title("Behaviour of cost function with iterations")
    plt.show()
    return;

# will plot the final result
def plotresult(X,y,theta):
    plt.plot(X,y,'rx',label='Training data')
    plt.axis([4,25,-5,25])
    plt.xlabel("Population of city in 10,000's")
    plt.ylabel("Profit in $10,000")
    x=np.linspace(4,25,200)
    data_x=x.reshape(200,1)
    add_ones=np.ones((200,1))
    data_x=np.hstack((add_ones,data_x))
    y=hypo(theta, data_x)
    line2, =plt.plot(x,y,label='linear Regression')
    plt.legend()
    plt.show()
    return;

def hypo(theta, X):                   # hypothesis function.
    h=X.dot(theta)                   # h(x)=X.theta.   theta=[theta1;theta2]
    return h;

def costfunc(theta,X,y):                    # function for computing cost function
    J=((X.dot(theta)-y).transpose()).dot(X.dot(theta)-y)      #vectorzired implementation of J
    (m,n)=X.shape
    J=J/(2*m);
    return J;

def gradientDescent(X,y,theta,alpha,iterations):
    (m,n)=X.shape
    J_val=np.linspace(0,0,iterations+1)       # finding the value of J after every step of gradientDescent
    for i in range(iterations+1):
        J_val[i]=costfunc(theta,X,y)
        theta=theta-(alpha/m)*((X.transpose()).dot((X.dot(theta)-y)))
    plotCost(J_val,iterations)
    return theta;

# visualizing data
plotdata(X_old,y)

X=np.hstack((add,X_old))                # adding a column of ones to X..
(m,n)=X.shape
#initializing parameters
initial_theta=np.zeros((n,1))                # will create a matrix of dimension 2x1

# checking whether algorithm is working correct or not.
print(costfunc(initial_theta,X,y))
print("the expected value is 32.07")

# gradient descent
theta=gradientDescent(X,y,initial_theta,alpha,iterations)
print("Theta found by gradient descent is")
print(theta)

# will plot the final result that is will plot the best fit.
plotresult(X_old,y,theta)


# using scipy.optimize...
y.shape=(m,)
theta2=op.minimize(fun=costfunc,x0=initial_theta,args=(X,y),method='TNC')
theta2=theta2.x
plotresult(X_old,y,theta)
