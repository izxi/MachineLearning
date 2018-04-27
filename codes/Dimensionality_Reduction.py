# Dimensionality Reduction: Feature Selection with VarianceThreshold and Univariance
#   VarianceThreshold is a useful tool to removing features with a threshold variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold()
# VarianceThreshold removes all zero-variance features by default. These features are any constant value features.
dataset = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
sel.fit_transform(dataset)
# change the threshold by adding threshold='threshold value' inside the brackets 
# during the instantiation of VarianceThreshold. 
# Where 'threshold value' is equal to p(1-p) Where 'p' is your threshold % in decimal format.
sel60 = VarianceThreshold(threshold=(0.6 * (1 - 0.6))) #  threshold of 60%  > variance of at least 60%
sel60.fit_transform(dataset)

# Univariance Feature Selection
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
import numpy as np 
import pandas
my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")
# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values
X = removeColumns(my_data, 0, 1) # row column dropped
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
# use the target function to obtain the Response Vector of my_data and store it as y
y = target(my_data, 1)
X.shape # take a look at X's shape  > (150, 4)
# chi2 is used as a univariance scoring function which returns p values.
# specified k=3 for the 3 best features to be chosen
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
X_new.shape # (150, 3)

# Feature Extraction with DictVectorizer and PCA
# DictVectorizer is a very simple Feature Extraction class as it can be used to
# convert feature arrays in a dict to NumPy/SciPy representations.
from sklearn.feature_extraction import DictVectorizer
dataset = [
...     {'Day': 'Monday', 'Temperature': 18},
...     {'Day': 'Tuesday', 'Temperature': 13},
...     {'Day': 'Wednesday', 'Temperature': 7},
... ]
vec = DictVectorizer() #  create an instance of DictVectorizer called vec
#  use the fit_transform function of vec with the parameter dataset and use the .toarray() on the final product
vec.fit_transform(dataset).toarray()
# dataset has been converted into an array format but pertaining its data. 
# can further review the data with the get_feature_names function of vec.
vec.get_feature_names() # ['Day=Monday', 'Day=Tuesday', 'Day=Wednesday', 'Temperature']
# use PCA to represent the data we used in feature selection(X_new) and 
# project it's dimensions so make sure you have completed that portion!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
%matplotlib inline

fig = plt.figure(1, figsize=(10, 8))  # create instances of plt.figure as fig and Axes3D as ax
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0) #  specificed dimensions, elev=0 and azim=0 to see the graph from where the z plane = 0.
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=y, cmap=plt.cm.spectral)  # plot X_new against y with the scatter function of ax
# create an instance of decomposition.PCA called pca with parameters of n_components=2.
pca = decomposition.PCA(n_components=2)
pca.fit(X_new)
# Use the transform function of pca with parameter X_new and equate it to a new variable called PCA_X. 
# This will be the projection resulting in the change of 3 features to 2.
PCA_X = pca.transform(X_new)

fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0)
ax.scatter(PCA_X[:, 0], PCA_X[:, 1], c=y, cmap=plt.cm.spectral)

PCA_X.shape # (150, 2) shape of PCA_X to show 2 features

# Elev controls the elevation of the z plane and azim controls the azimuth angle in the x,y plane.
