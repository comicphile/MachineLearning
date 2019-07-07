# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:21:32 2019

@author: imkri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values #independent features
y=dataset.iloc[:,1]



#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#Simple Linear Regression

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test results
y_pred=regressor.predict(X_test)

#Visualizing the test set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience [Training Set]")
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
plt.show()


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience [Test Set]")
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
plt.show()