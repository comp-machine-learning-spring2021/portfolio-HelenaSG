# K-means and Spectual Clustering from scratch

Contents:
1. [Write my own k-means](#My k-means implementation)
2. Pre-process data for Spectral clustering
3. Build and deploy spectral clustering
4. Perform k-means and spectral clustering on an "interesting" data set
5. Compare my implementions with the built in `sklearn` kmeans functions, discuss the efficacy of the work via time

*Notes: *
*The implementations for 1-3 are in ce1.py, while the plots and discussions for 4-5 are in ce1-notes.ipynb.*

## 1: My k-means implementation
This is a "cold" k-means implementation and include my justification for the stopping condition(s) that I used in my implementations. 
The implementation is called `my_kmeans()` and set at three inputs **in this 
order** -
* A numpy array
* The number of cluster (ie. _k_)
* The random_state

The implementation terminates with output including:
- The cluster centers
- Cluster labels for the data points


## 2: Pre-process data for Spectral clustering
Spectral clustering relies on local pairwise relationships for the clustering. As such, we need to pre-process our data appropriately. 

### Part A: Create an adjacency matrix for the data
The first step in the pre-processing is to create an _adjacency_ matrix **A** that 
signals which points are near each other. There are several ways to do this, we will 
use the _E_-neighborhood version. That is, any pair of points that are within a 
certain distance of each other, we say are "close", and any pair of points that are 
not within that distance are "far". 

To create the adjacency matrix, compute the pairwise distances between each pair of 
data points. Next, turn this matrix into a _binary_ matrix, that is one with only 
entries equal to 1 and 0. For this transformation, if the pairwise distance between 
the _i_-th datapoint and the _j_-th datapoint is less than 1/2, then set the 
_i,j_-th entry of **A** to 1. If it is not less than 1/2, then set _i,j_-th entry 
of **A** to 0. 

In both cases, we assume that the _i_-th datapoint and the _j_-th datapoint are not 
the same datapoint. For the _i,i_-th entry of **A**, set this also to 0. (Letting it 
be non-zero says that there is a "loop" between the point and itself). 

The function `make_adj()` takes a numpy array as input and return a 2D numpy array with only 1 and 0 entries. 

### Part B: Create a Laplacian for the adjacency matrix
There are a few ways that one can do spectral clustering. The three in the classic 
[_A Tutorial on Spectral Clustering_ by Ulrike von Luxburg](https://arxiv.org/pdf/0711.0189.pdf)
rely on different versions of the graph Laplacian, which is a discretized version 
of the Laplacians used to decribe fluid flow. 

Our Adjacency matrix denotes when points are near to each other with a 1 and those 
that are far with a 0. The Laplacian wants to mimic how information might "flow" 
between these data points over their "close connections". 

Part of each of the common Laplacian formulations is the _degree matrix_ **D** which 
simply counts the number of datapoints that each data point is near. The degree 
matrix **D** is a diagonal matrix where the _i,i_ entry equals the number of points 
that the _i_-th datapoint is near to. I find the `diag` command to be helpful here. 

In this part, we compute the unnormalized Laplacian: **L** = **D** - **A**. 
The function `my_laplacian()` takes an adjacency matrix (as a numpy array) 
as input and returns the unnormalized Laplacian also as a numpy array. 

## 3: Perform a special dimension reduction, and then k-means

**Spectral Clustering** combines dimension reduction and k-means.

With our unnormalized Laplacian computed, we are ready for one version of 
spectral clustering. Simplistically, spectral clustering is simply a 
dimension reduction followed by k-means. For this data, we are looking for two 
clusters, so _k_ = 2. 

Spectral clustering on the unnormalized Laplacian **L** is:
* Computing the eigenvectors of **L**
* Order the eigenvalues from smallest to greatest, and place the eigenvectors 
  in the same order
* Identify the first **non-zero** (ie. above machine tolerance) eigenvalue. The 
  eigenvector associated to this eigenvalue is called the _first_ eigenvector. 
* Select the first _k_ eigenvectors 
* Compute k-means on the selected eigenvectors

The function `spect_clustering()` takes an unnormalized Laplacian and 
the number of clusters as input and returns both the labels and the centers of 
the resulting clusters. 

## 4: Apply K-means and Spectual Clustering
In the jupyter notebook `ce1-notes.ipynb`, I used my implementations `my_kmeans()` and `spect_clustering()` on an intereting dataset. 

## 5: Comparison and Discussion
This part involves a discussion about the efficacy of the work via time (in seconds).
I created two dimensional plots showing my k-means and spectual clustering results, paired with the results of the corresponding built in `sklearn` function. 

Let's evaluate these results in the jupyter notebook! 

#### Resources consulted
0. [_A Tutorial on Spectral Clustering_ by Ulrike von Luxburg](https://arxiv.org/pdf/0711.0189.pdf) 
   Note this is **the classic** text in spectral clustering. 
1. [Spectral Clustering Algorithm Implemented From Scratch](https://towardsdatascience.com/unsupervised-machine-learning-spectral-clustering-algorithm-implemented-from-scratch-in-python-205c87271045)
2. [How to generate random points in a circular distribution](https://stackoverflow.com/questions/30564015/how-to-generate-random-points-in-a-circular-distribution)
3. [rand helpfile](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.rand.html)
4. [unique helpfile in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html)
5. [SO - threshold in 2D numpy array](https://stackoverflow.com/questions/36719997/threshold-in-2d-numpy-array/36720130)
6. [tril in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tril.html#numpy.tril)
7. [triu in numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.triu.html)
