# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:19:25 2020

@author: Vaibhav Rastogi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#linear regression model
from sklearn.linear_model import LinearRegression
linregressor=LinearRegression()
linregressor.fit(X,y)
#y1=linregressor.predict([[6.5]])

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

linreg2=LinearRegression()
linreg2.fit(X_poly,y)
#y2=linreg2.predict([[6.5]])


#Linear
plt.scatter(X,y,color='red')
plt.plot(X,linregressor.predict(X),color='blue')
plt.title('Truth vs Bluff lInear Reg')
plt.xlabel('Pos lEVEL')
plt.ylabel('Salary')
plt.show()

#Poly
plt.scatter(X,y,color='red')
plt.plot(X,linreg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth vs Bluff Poly Reg')
plt.xlabel('Pos lEVEL')
plt.ylabel('Salary')
plt.show()


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linreg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
y1=linregressor.predict([[6.5]])

# Predicting a new result with Polynomial Regression
y2=linreg2.predict(poly_reg.fit_transform([[6.5]]))

