import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def scale_feartures(features):
    """transform data such that its distribution has a mean value 0 
       and standard deviation of 1

    Parameters
    ----------
    features : DataFrame 
        the feature columns needed to be scaled

    Returns
    -------
    Numpy array
        scaled data in form of a numpy array
    """
    
    scaler = StandardScaler().fit(features)
    scaled_features = scaler.transform(features)
    
    return scaled_features

def get_cv_scores(clfs, X, y):
    """conduct cross-validation on the classifiers 

    Parameters
    ----------
    clfs : list
        a list of pre-defined sklearn classifiers
    X : DataFrame 
        a set of input variables
    y : Series
        a column of output variable

    Returns
    -------
    a list 
        cv-scores of each model
    """
    
    scores_lst = []
    cv = KFold(n_splits=6)
    
    for clf in clfs:
        scores = cross_val_score(clf, X, y, cv=cv)
        scores_lst.append(scores.mean())
        
    return scores_lst


    
