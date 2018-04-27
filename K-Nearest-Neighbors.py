

import numpy as np 
import pandas 
from sklearn.neighbors import KNeighborsClassifier

# Using my_data as the skulls.csv data read by panda,
# declare variables X as the Feature Matrix (data of my_data) and
# y as the response vector (target)
my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")
#	  Unnamed: 0	epoch	    mb	  bh	  bl	nh
# 0	1	          c4000BC	  131	  138	  89	49
# 1	2	          c4000BC	  125	  131	  92	48
# ...
# Use the target function for the response vector
def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)
# the removeColumns function for the Feature Matrix
#   Remove the column containing the target name since it doesn't contain numeric values.
#   Also remove the column that contains the row number
#   axis=1 means we are removing columns instead of rows.
#   Function takes in a pandas array and column numbers and returns a numpy array without the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

X = removeColumns(my_data, 0, 1)
y = target(my_data, 1)

# sklearn: split the X and y into two different sets: The training and testing set.
from sklearn.cross_validation import train_test_split
# The X and y are the arrays required before the split,
# the test_size represents the ratio of the testing dataset,
# and the random_state ensures we obtain the same splits.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=7)

print X_trainset.shape  # (105, 4)
print y_trainset.shape  # (105,)
print X_testset.shape   # (45, 4)
print y_testset.shape   # (45,)

# create declarations of KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh23 = KNeighborsClassifier(n_neighbors = 23)
neigh90 = KNeighborsClassifier(n_neighbors = 90)

# fit each instance of KNeighborsClassifier with the X_trainset and y_trainset
neigh.fit(X_trainset, y_trainset)
neigh23.fit(X_trainset, y_trainset)
neigh90.fit(X_trainset, y_trainset)
# Let's pass the y_testset in the predict function each instance of KNeighborsClassifier 
# but store it's returned value into pred, pred23, pred90 (corresponding to each of their names)
pred = neigh.predict(X_testset)
pred23 = neigh23.predict(X_testset)
pred90 = neigh90.predict(X_testset)
# compute neigh's prediction accuracy
from sklearn import metrics
print("Neigh's Accuracy: "), metrics.accuracy_score(y_testset, pred)  # Neigh's Accuracy:  0.2222222222222222
print("Neigh23's Accuracy: "), metrics.accuracy_score(y_testset, pred23)  # Neigh23's Accuracy:  0.24444444444444444
print("Neigh90's Accuracy: "), metrics.accuracy_score(y_testset, pred90)  # Neigh90's Accuracy:  0.13333333333333333

# As shown, the accuracy of neigh23 is the highest. 
# When n_neighbors = 1, the model was overfit to the training data (too specific) and
# when n_neighbors = 90, the model was underfit (too generalized). In comparison, 
# n_neighbors = 23 had a good balance between Bias and Variance, 
# creating a generalized model that neither underfit the data nor overfit it.


# Train/Test Split on the Diabetes Dataset with Linear Regression
from sklearn.datasets import load_diabetes 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
%matplotlib inline
# create an instance of the diabetes data set by using the load_diabetes function
diabetes = load_diabetes()
diabetes_X = diabetes.data[:, None, 2]
# create an instance of the LinearRegression called LinReg
LinReg = LinearRegression()

X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)
# Train the LinReg model using X_trainset and y_trainset
LinReg.fit(X_trainset, y_trainset)
# Use plt's scatter function to plot all the datapoints of X_testset and y_testset and color it black
plt.scatter(X_testset, y_testset, color='black')
# Use plt's plot function to plot the line of best fit with X_testset and LinReg.predict(X_testset). Color it blue with a linewidth of 3.
plt.plot(X_testset, LinReg.predict(X_testset), color='blue', linewidth=3)
