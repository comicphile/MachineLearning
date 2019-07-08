# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:12:59 2019

@author: imkri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values #independent features
y=dataset.iloc[:,2:3].values


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
"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)



#Fitting Regression Model to the dataset
from sklearn.svm  import SVR
regressor=SVR(kernel="rbf")
regressor.fit(X,y)


#Predicting a new result with Polynomial Regression
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing Regression Results
plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Salary vs Level [SV Regression]")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Visualizing Regression Results(for higher resolution)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Salary vs Level [SV Regression]")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()