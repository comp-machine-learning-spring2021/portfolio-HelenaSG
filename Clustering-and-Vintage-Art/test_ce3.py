import pytest
import pandas as pd
import numpy as np
import ce3

my_data = np.loadtxt("testdata.csv", delimiter = ",")

# Tests for k-means clustering implementation

def test_my_kmeans_type():
	assert isinstance(ce3.my_kmeans(my_data, 2, 1), tuple)

def test_my_kmeans_shape():
	expected = 2
	assert len(ce3.my_kmeans(my_data, 2, 1)) == expected

def test_my_kmeans_center_num():
	expected = (2,2)
	centers_shape = ce3.my_kmeans(my_data, 2, 1)[0].shape
	assert centers_shape == expected
	
def test_my_kmeans_labels():
	expected = 1
	label_max = np.max(ce3.my_kmeans(my_data, 2, 1)[1])
	assert label_max == expected

def test_different_cols():
	expected = False
	centers=ce3.my_kmeans(my_data, 2, 1)[0]
	comp_cols = sum(centers[:,0] == centers[:,1]) == 2
	assert comp_cols == expected
    
# Tests for spectual clustering implementation

def test_make_adj_size():
	expected = (1000,1000)
	assert ce3.make_adj(my_data).shape == expected

def test_make_adj_diag():
	# Test for empty diagonal
	expected = 0
	out = ce3.make_adj(my_data)
	assert np.sum(np.diag(out)) == expected

def test_make_adj_values():
	# Test that the matrix is binary
	expected = [0, 1]
	out = ce3.make_adj(my_data)
	assert list(np.unique(out)) == expected

def test_my_laplacian_size():
	expected = (1000,1000)
	AM = ce3.make_adj(my_data)
	assert ce3.my_laplacian(AM).shape == expected

def test_my_laplacian_diag():
	# Test that the degree matrix diagonal is non-negative
	expected = 1000
	AM = ce3.make_adj(my_data)
	out = ce3.my_laplacian(AM)
	assert np.sum(np.diag(out)>=0) == expected

def test_my_laplacian_else():
	expected = 1000*1000
	AM = ce3.make_adj(my_data)
	out = ce3.my_laplacian(AM)
	#Consider the top triangular half of the matrix
	upt_out = np.triu(out,1)
	#Consider the lower triangular half of the matrix
	downt_out = np.tril(out,-1)
	test_mat = upt_out + downt_out
	assert np.sum(test_mat<=0) == expected

def test_spect_clustering_type():
	AM = ce3.make_adj(my_data)
	L = ce3.my_laplacian(AM)
	out = ce3.spect_clustering(L,7)
	assert isinstance(out, tuple)

def test_spect_clustering_shape():
	expected = 2
	AM = ce3.make_adj(my_data)
	L = ce3.my_laplacian(AM)
	out = ce3.spect_clustering(L,7)
	assert len(out) == expected

def test_spect_clustering_labels():
	expected = 6
	AM = ce3.make_adj(my_data)
	L = ce3.my_laplacian(AM)
	out = ce3.spect_clustering(L,7)
	label_max = np.max(out[0])
	assert label_max == expected

def test_spect_clustering_center_num():
	expected = (7,7)
	AM = ce3.make_adj(my_data)
	L = ce3.my_laplacian(AM)
	out = ce3.spect_clustering(L,7)
	centers_shape = out[1].shape
	assert centers_shape == expected
