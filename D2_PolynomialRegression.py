# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:49:08 2019

@author: imkri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values #independent features
y=dataset.iloc[:,2]

"""
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0]) #Dummy Encoding
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""


"""#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Fitting Linear Regression Model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression Model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing Linear Regression Results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Salary vs Level [Simple Linear Regression]")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Visualizing Polynomial Regression Results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Salary vs Level [Polynomial Regression]")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
