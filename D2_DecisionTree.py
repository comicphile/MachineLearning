# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:30:13 2019

@author: imkri
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values #independent features
y=dataset.iloc[:,2].values

#Fitting Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#Predicting a new result with Polynomial Regression
y_pred=regressor.predict(6.5)


#Visualizing Regression Results(for higher resolution)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Salary vs Level [Decision Tree Regression]")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
