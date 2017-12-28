# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
one=OneHotEncoder(categorical_features=[3])
X=one.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap We only take n-1 dummy variables of n dummy.
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

# Predicting the Test set results
y_pred=lr.predict(X_test)


# building a optimal model using backward elmination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)                   # axis =1 represents columns

# selecting variables        sl=0.05
X_opt=X[:,[0,3,5]]
ols=sm.OLS(endog=y,exog=X_opt).fit()
ols.summary()                   # highest p value is 2 and p>sl


