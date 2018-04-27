# Train/Test Split on the Diabetes Dataset with Linear Regression
from sklearn.datasets import load_diabetes 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
%matplotlib inline
diabetes = load_diabetes()
diabetes_X = diabetes.data[:, None, 2]
LinReg = LinearRegression()
X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)
LinReg.fit(X_trainset, y_trainset)
plt.scatter(X_testset, y_testset, color='black')
plt.plot(X_testset, LinReg.predict(X_testset), color='blue', linewidth=3)
# Supervised Learning - Regression
from sklearn.datasets import load_diabetes 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
%matplotlib inline # 4notebook
diabetes = load_diabetes() 
diabetes_X = diabetes.data[:, None, 2]
LinReg = LinearRegression()
LinReg.fit(diabetes_X, diabetes.target)
plt.scatter(diabetes_X, diabetes.target, color='black')
plt.plot(diabetes_X, LinReg.predict(diabetes_X), color='blue', linewidth=3)
# Regression Evaluation Metrics
print(np.mean(abs(LinReg.predict(X_testset) - y_testset))) # MAE
print(np.mean((LinReg.predict(X_testset) - y_testset) ** 2) ) # MSE
print(np.mean((LinReg.predict(X_testset) - y_testset) ** 2) ** (0.5) ) # RMSE



# K-means Clustering
## Generating Random Data
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
%matplotlib inline
# numpy's random.seed() function, where the seed will be set to 0 
np.random.seed(0)
# Input
#   n_samples: The total number of points equally divided among clusters. Value will be: 5000
#   centers: The number of centers to generate, or the fixed center locations.  Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
#   cluster_std: The standard deviation of the clusters.  Value will be: 0.9
# Output
#   X: Array of shape [n_samples, n_features]. (Feature Matrix)   The generated samples.
#   y: Array of shape [n_samples]. (Response Vector)    The integer labels for cluster membership of each sample.
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.') # Display the scatter plot of the randomly generated data.
## Setting up K-Means
# init: Initialization method of the centroids.
#   Value will be: "k-means++"
#   k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
#   Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
#         The final results will be the best output of n_init consecutive runs in terms of inertia.
#   Value will be: 12
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
## Creating the Visual Plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot with a black background (background is black because we can see the points
# connection to the centroid.
ax = fig.add_subplot(1, 1, 1, axisbg = 'black')
# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each data point is in.
for k, col in zip(range(len([[2, 2], [-2, -1], [4, -3], [1, 1]])), colors):
    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are labeled as false.
    my_members = (k_means_labels == k)
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans') # Title of the plot
ax.set_xticks(()) # Remove x-axis ticks
ax.set_yticks(()) # Remove y-axis ticks
plt.show() # Show the plot
# Display the scatter plot from above for comparison.
plt.scatter(X[:, 0], X[:, 1], marker='.')
## Unsupervised Learning - K-means Clustering
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_iris
np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
iris = load_iris()
X = iris.data 
y = iris.target
# see what K-Means produces
estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}

fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()
