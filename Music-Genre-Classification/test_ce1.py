import pytest
import ce1
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

tracks = pd.read_csv('songs.csv')
echonest_metrics = pd.read_json('echonest-metrics.json', precise_float=True)
df = pd.merge(left=echonest_metrics, right=tracks[['track_id', 'genre_top']], on='track_id')
df.loc[df['genre_top'] == 'Rock', 'genre'] = 0
df.loc[df['genre_top'] == 'Hip-Hop', 'genre'] = 1
features = ['speechiness', 'danceability', 'instrumentalness']

def test_scale_feartures_shape():
    in_df = df[features] 
    out = ce1.scale_feartures(in_df)
    assert out.shape == in_df.shape
    
def test_scale_feartures_type():
    in_df = df[features] 
    out = ce1.scale_feartures(in_df)
    assert isinstance(out, np.ndarray)
    
def test_scale_feartures_mean():
    in_df = df[features] 
    out = ce1.scale_feartures(in_df)
    assert np.isclose(np.sum(np.mean(out, axis=0)),0)
    
def test_scale_feartures_std():
    in_df = df[features] 
    out = ce1.scale_feartures(in_df)
    assert np.isclose(np.all(np.std(out, axis=0)),1)

def test_get_cv_scores_type():
    X = df[features] 
    y = df['genre'] 
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='rbf', C=6, random_state=1)
    rf = RandomForestClassifier()
    clfs = [knn,svm,rf]
    out = ce1.get_cv_scores(clfs, X, y)
    assert all(isinstance(element, float) for element in out)
      
def test_get_cv_scores_shape():
    X = df[features] 
    y = df['genre'] 
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='rbf', C=6, random_state=1)
    rf = RandomForestClassifier()
    clfs = [knn,svm,rf]
    out = ce1.get_cv_scores(clfs, X, y)
    assert len(out) == 3
