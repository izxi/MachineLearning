# Agglomerative Hierarchical Clustering

import numpy as np 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
%matplotlib inline

# Generating Random Data

#   Input these parameters into make_blobs:
#     n_samples: The total number of points equally divided among clusters.
#       Choose a number from 10-1500
#     centers: The number of centers to generate, or the fixed center locations.
#       Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]])
#     cluster_std: The standard deviation of the clusters. The larger the number, the further apart the clusters
#       Choose a number between 0.5-1.5
#   Save the result to X2 and y2.
X2, y2 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
plt.scatter(X2[:, 0], X2[:, 1], marker='.') 

# Agglomerative Clustering

#   n_clusters: The number of clusters to form as well as the number of centroids to generate.
#     Value will be: 4
#   linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
#     Value will be: 'complete'
#     Note: It is recommended you try everything with 'average' as well
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglom.fit(X2,y2)

plt.figure(figsize=(6,4)) # Create a figure of size 6 inches by 4 inches.
# These two lines of code are used to scale the data points down, Or else the data points will be scattered very far apart.
x_min, x_max = np.min(X2, axis=0), np.max(X2, axis=0) # Create a minimum and maximum range of X2.
X2 = (X2 - x_min) / (x_max - x_min) # Get the average distance for X2.
# This loop displays all of the datapoints.
for i in range(X2.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X2[i, 0], X2[i, 1], str(y2[i]),
             color=plt.cm.spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
plt.axis('off')
# Display the plot
plt.show()
# Display the plot of the original data before clustering
plt.scatter(X2[:, 0], X2[:, 1], marker='.')

# Dendrogram Associated for the Agglomerative Hierarchical Clustering

dist_matrix = distance_matrix(X2,X2) # the distance values are symmetric, with a diagonal of 0's
Z = hierarchy.linkage(dist_matrix, 'complete') # 'complete' for complete linkage
dendro = hierarchy.dendrogram(Z)  


