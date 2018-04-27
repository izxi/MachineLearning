# Support Vector Regression which is a type of regression from SVM (Support Vector Machine). 

from sklearn.svm import SVR
import numpy as np 
# set the random seed to ensure the same results with the random.seed function of np with parameter 5
np.random.seed(5)
X = np.sort(10 * np.random.rand(30, 1), axis=0)
y = np.sin(X).ravel()
# SVR uses: rbf, linear, and sigmoid, C=1e3 (1 * 10^3) which is the penalty parameter of the error term
# RBF short for Radial basis function
svr_rbf = SVR(kernel='rbf', C=1e3)
svr_linear = SVR(kernel='linear', C=1e3)
svr_sigmoid = SVR(kernel='sigmoid', C=1e3)

svr_rbf.fit(X,y)
svr_linear.fit(X,y)
svr_sigmoid.fit(X,y)

y_pred_rbf = svr_rbf.predict(X)
y_pred_linear = svr_linear.predict(X)
y_pred_sigmoid = svr_sigmoid.predict(X)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
%matplotlib inline

plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_pred_rbf, c='g', label='RBF model')
plt.plot(X, y_pred_linear, c='r', label='Linear model')
plt.plot(X, y_pred_sigmoid, c='b', label='Sigmoid model')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

