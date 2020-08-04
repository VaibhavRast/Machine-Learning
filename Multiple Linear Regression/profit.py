# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:18:59 2020

@author: Vaibhav Rastogi
"""

#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#avoid dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

#Build optimal model using backward elimination
#import statsmodels.formula.api as smf
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

#Start Backward elimination
X_opt=X[:,[0,1,2,3,4,5]]
sl=0.05
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

def backwardElimination(x, sl): 
    numVars = len(x[0]) 
    for i in range(0, numVars): 
        regressor_OLS = sm.OLS(y, x).fit() 
        maxVar = max(regressor_OLS.pvalues).astype(float) 
        if maxVar > sl: 
            for j in range(0, numVars - i): 
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): 
                    x = np.delete(x,j,1)
    regressor_OLS.summary() 
    return x

X_model=backwardElimination(X_opt,sl)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_model, y, test_size = 0.2, random_state = 0)

#Fit the model

regressor=LinearRegression()
regressor.fit(X_train1,y_train1)

y_pred1=regressor.predict(X_test1)
    





