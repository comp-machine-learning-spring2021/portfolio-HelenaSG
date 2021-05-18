# Vintage Picture Generator
    “Art is the journey of a free soul.” - Alev Oguz

## :art: Introduction
Art is everywhere in life. In my mind, photography is a form of art. I have a great interest in image post-processing because I think as I drastically change the outcome of the image captured by a camera, I’m manipulating it to fit my artistic expression. There are many ways of post-processing, the most common tool is Adobe Photoshop, and the easiest way is to apply a filter. We can make the photo look retro, have a color tint, or apply some other kind of effect. I am a huge fan of vintage styles, but simply adjusting some parameters or using some filters sometimes does not give me the vintage effect I’m looking for.

I wanted my version of vintage pictures to have a simple color scheme, and look a bit surreal and 2D. How can I get such transformations? This is how I came up with the K-means clustering algorithm. To get a better idea of how it works, I first implemented it from scratch. I also investigated its limitations by comparing it with another clustering algorithm called Spectral clustering. In the end, I applied the K-means algorithm to conduct vintage image transformations.

:small_blue_diamond: **An in-Depth Exploration of Clustering Algorithms:** 

  * K-means clustering and Spectral clustering from scratch
  * Comparing my implementions with the built in sklearn kmeans functions
  * Comparing two clustering algorithms 
  * Discussing the efficacy of the work via time
  * An artistic application of k-means

## :art: K-means and Spectral clustering implementations

The goal of k-means is to group individual data points into **k** clusters. At the beginning of k-means, we input our dataset and **k**, the number of clusters that we believe exist in the data. The first step is to initialize k centroids (randomly), then, we 1)assign each datapoint to the closest center, 2)each subset of the datapoints assigned to each center forms a cluster, 3)for each cluster, re-compute the center. We repeat steps 1-3 untill convergence, stopping condition, or maximum iterations.

Spectral clustering combines dimension reduction and k-means. The first step in the pre-processing is to create an _adjacency_ matrix **A** that signals which points are near each other. Specifically, I used the _E_-neighborhood version (denotes 1 if is within a certain distance and 0 otherwise). The second step is to create a Laplacian for the adjacency matrix. The Laplacian wants to mimic how information might "flow" between these data points over their "close connections".  Specifically, it means to compute _unnormalized Laplacian_ **L** = _degree matrix_ **D** - _adjacency matrix_ **A**. With the unnormalized Laplacian computed, we can go ahead and compute the eigenvectors of L, select the first k eigenvectors, and finally, compute k-means on the selected eigenvectors.

The code for the implementations can be found [here](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Clustering-and-Vintage-Art/ce3.py).

## :art: Comparisons and time analysis

When we implement the spectral clustering, we do some additional steps like providing an adjacency matrix and computing the eigenvectors. And eventually, we have K-means clustering as the final step of Spectral clustering. That’s why Spectral clustering takes longer to run and gives different results than K-means. Given that spectral clustering is just some pre-processing steps + K-means, we can use those steps with any clustering algorithm to get better results on the dataset when other algos are not ideal to use.

The full discussion for this part can be found [here](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Clustering-and-Vintage-Art/Different-Clustering-Algorithms.ipynb).  

## :art: Application: transforming pictures to vintage styles 

In the context of image data, the role of K-means clustering is essentially color reduction. As the first step, we read the image file from the local directory. For a color image, the function imread() returns a three-dimensional array, where each column corresponds to a color - the order is BGR (blue, green, red). When using K-means clustering on this image data, each cluster centroid represents a cluster of pixels. That's to say, the number of colors of the final picture is equal to the number of clusters (k) defined when applying K-means, and the centroids decide what the exact colors are in the color palette. Once we have the centroids, we can recreate the image by replacing the color of each pixel with the color of its cluster centroid. By setting the k to 4, we will have the original photo transformed into one with only four colors. The jupyter notebook for this part can be found [here](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Clustering-and-Vintage-Art/Vintage-Picture-Generator.ipynb). 

The following is a gallery of my creations :) with the original image on the lefthand side and its result on the right.
![gallery](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Clustering-and-Vintage-Art/gallery.jpeg)

