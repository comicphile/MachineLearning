# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:37:24 2019

@author: imkri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values #independent features
y=dataset.iloc[:,4]



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3]) #Dummy Encoding
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X=X[:,1:]



#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

#Since, x2 has highest p-value which is greater than 0.05, thus we should remove it
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()
