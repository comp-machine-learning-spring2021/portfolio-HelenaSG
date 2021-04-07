import pytest
import pandas as pd
import numpy as np
import ce1

students = pd.read_csv("students_info.csv")
justtwo_np = students[["sleep","coffee"]].to_numpy()

my_data = np.loadtxt("testdata.csv", delimiter = ",")

# Tests for k-means clustering implementation

def test_my_kmeans_type():
	assert isinstance(ce1.my_kmeans(justtwo_np, 6, 2019), tuple)

def test_my_kmeans_shape():
	expected = 2
	assert len(ce1.my_kmeans(justtwo_np, 6, 2019)) == expected

def test_my_kmeans_center_num():
	expected = (6,2)
	centers_shape = ce1.my_kmeans(justtwo_np, 6, 2019)[0].shape
	assert centers_shape == expected
	
def test_my_kmeans_labels():
	expected = 5
	label_max = np.max(ce1.my_kmeans(justtwo_np, 6, 2019)[1])
	assert label_max == expected

def test_different_cols():
	expected = False
	centers=ce1.my_kmeans(justtwo_np, 6, 2019)[0]
	comp_cols = sum(centers[:,0] == centers[:,1]) == 6
	assert comp_cols == expected
    
# Tests for spectual clustering implementation

# Part A
def test_make_adj_size():
	expected = (1000,1000)
	assert ce1.make_adj(my_data).shape == expected

def test_make_adj_diag():
	# Test for empty diagonal
	expected = 0
	out = ce1.make_adj(my_data)
	assert np.sum(np.diag(out)) == expected

def test_make_adj_values():
	# Test that the matrix is binary
	expected = [0, 1]
	out = ce1.make_adj(my_data)
	assert list(np.unique(out)) == expected

# Part B
def test_my_laplacian_size():
	expected = (1000,1000)
	AM = ce1.make_adj(my_data)
	assert ce1.my_laplacian(AM).shape == expected

def test_my_laplacian_diag():
	# Test that the degree matrix diagonal is 
	#    non-negative
	expected = 1000
	AM = ce1.make_adj(my_data)
	out = ce1.my_laplacian(AM)
	assert np.sum(np.diag(out)>=0) == expected

def test_my_laplacian_else():
	expected = 1000*1000
	AM = ce1.make_adj(my_data)
	out = ce1.my_laplacian(AM)
	#Consider the top triangular half of the matrix
	upt_out = np.triu(out,1)
	#Consider the lower triangular half of the matrix
	downt_out = np.tril(out,-1)
	test_mat = upt_out + downt_out
	assert np.sum(test_mat<=0) == expected

def test_spect_clustering():
	AM = ce1.make_adj(my_data)
	L = ce1.my_laplacian(AM)
	out = ce1.spect_clustering(L,7)
	assert isinstance(out, tuple)

def test_spect_clustering_shape():
	expected = 2
	AM = ce1.make_adj(my_data)
	L = ce1.my_laplacian(AM)
	out = ce1.spect_clustering(L,7)
	assert len(out) == expected

def test_spect_clustering_labels():
	expected = 6
	AM = ce1.make_adj(my_data)
	L = ce1.my_laplacian(AM)
	out = ce1.spect_clustering(L,7)
	label_max = np.max(out[0])
	assert label_max == expected

def test_spect_clustering_center_num():
	expected = (7,7)
	AM = ce1.make_adj(my_data)
	L = ce1.my_laplacian(AM)
	out = ce1.spect_clustering(L,7)
	centers_shape = out[1].shape
	assert centers_shape == expected
