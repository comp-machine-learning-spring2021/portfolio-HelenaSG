import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Helper function that recomputes the centers
def updateCentroids(data, cluster, centroids):
    """
    inputs
    ----------
        numpy array: The dataset
        numpy array: Cluster labels
        numpy array: Centroids
    
    returns
    -------
        numpy array: Centroids
    """
    array = []
    # for each subset of the data that is in that cluster
    for k in range(centroids.shape[0]):
        # compute the average
        newCentroid = data[cluster==k].mean(axis=0)
        array.append(newCentroid)
    return np.array(array)

# Main K-means clustering
def my_kmeans(numpyArray, clusterNum, randomState):    
    """
    inputs
    ----------
        numpy array: The dataset
        integer: The number of cluster (ie. k)
        integer: The random_state
    
    returns
    -------
        numpy array: Centroids
        numpy array: Cluster labels
    """

    # Initialize clusters' centers
    centroids = pd.DataFrame(numpyArray).sample(clusterNum, random_state = randomState).to_numpy()

    # Repetitive steps
    # Termination condition: when the centroids do not change,
    # or when the number of iterations exceeds 1000.

    for i in range(1,1000):

        # Assign each data point to the closest center
        dists = distance.cdist(numpyArray, centroids, 'euclidean')
        clusters = np.argmin(dists, axis=1)

        newCentroids = updateCentroids(numpyArray, clusters, centroids)

        if np.array_equal(newCentroids, centroids):
            break
            
        newCentroids = centroids
    
    return newCentroids, clusters

# Spectral clustering

# Helper function that computes k-means
def full_kmeans(data, k):
    """
    inputs
    ----------
        numpy array: The dataset
        integer: The number of cluster (ie. k)
        integer: The random_state
        
    returns
    ----------
        numpy array: Cluster labels
        list: Centroids
    """

    km_alg = KMeans(n_clusters=k, init="random", random_state=1, max_iter=200)
    fit = km_alg.fit(data)
    labels = fit.labels_
    centers = fit.cluster_centers_

    return labels,centers

# Helper function that creates an adjacency matrix for the data
def make_adj(np_array):
    """
    inputs
    ----------
        numpy array: The dataset  
        
    returns
    ----------
        numpy array: A binarized matrix  
    """

    matrix = distance.cdist(np_array, np_array, 'euclidean')

    matrix = np.where(matrix >= 0.5, 2, matrix)
    matrix = np.where(matrix == 0, 2, matrix)

    # if less than 1/2, set to 1
    matrix = np.where(matrix < 0.5, 1, matrix)

    # if not less than 1/2, or same datapoint (== 0), set to 0
    matrix = np.where(matrix == 2, 0, matrix)
    
    return matrix

# Helper function that creates a Laplacian for the adjacency matrix
def my_laplacian(np_array):
    """
    inputs
    ----------
        numpy array: Adjacency matrix  
        
    returns
    ----------
        numpy array: Unnormalized Laplacia 
    """
    
    n = np_array.shape[0]

    # initialize values of degree for each i-th datapoint
    degree = np.zeros(n)

    # sum of degree for each row = nearby points for i-th datapoints
    rowsum = np_array.sum(axis=1)
    for i in range(0, n):
        degree[i] = rowsum[i]

    D = np.diag(degree)

    # L = D - A
    unnormalized_Laplacian = D - np_array

    return unnormalized_Laplacian

# Main Spectual Clustering
def spect_clustering(L,k):
    """
    inputs
    ----------
        numpy array: unnormalized Laplacian
        
    returns
    ----------
        numpy array: Cluster labels 
        list: Centroids
    """   

    # Compute the eigenvectors of L
    eig_vals, eig_vecs = np.linalg.eig(L)

    # Order the eigenvalues from smallest to greatest
    inds = (np.abs(eig_vals)).argsort()
    eig_vals = eig_vals[inds]
    # ... and place the eigenvectors in the same order
    pri_comps = eig_vecs[:,inds]

    # Identify the first non-zero eigenvalue
    tofind = eig_vals!=0
    non_zero_index = np.where(tofind.any(axis=0), tofind.argmax(axis=0), -1)

    # Select the first k eigenvectors
    selected_comps = pri_comps[:, non_zero_index:k]

    # Compute k-means on the selected eigenvectors
    km = full_kmeans(selected_comps, k)
    labels = km[0]
    centers = km[1]

    return labels, centers
