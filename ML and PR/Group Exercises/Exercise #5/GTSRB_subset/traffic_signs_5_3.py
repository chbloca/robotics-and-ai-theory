# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:48:51 2019

@author: hehu
"""

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from math import floor, ceil
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from simplelbp import local_binary_pattern
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score


cvn = 3

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)


def paramenter_searcher_manual(F,y):
    
    X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.2)

    clf_list = [LogisticRegression(), SVC()]
    clf_name = ['LR', 'SVC']
    
    C_range = np.arange(0.00001,1,0.1)
    best_score = []
    C_result = []
    penalty_final = []

    for clf,name in zip(clf_list, clf_name):
        for C in C_range:
            for penalty in ["l1", "l2"]:
                clf.C = C
                clf.penalty = penalty
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                C_result.append(C)
                best_score.append(score)
                penalty_final.append(penalty)
                
    index = np.argmax(best_score)
    
    
    return best_score[index], C_result[index], penalty_final[index]

def parameter_searcher_grid(F,y):
    X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.2)

    svc = SVC(gamma="scale")
    
    param_grid = [
      {'C': [0.00001,1], 'kernel': ['linear']},
      {'C': [0.00001,1], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    
    clf = GridSearchCV(svc, param_grid, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)
    #score = accuracy_score(y_test, y_pred)
    sorted(clf.cv_results_.keys)
    

def parameter_searcher_grid_rand(F,y):
    X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.2)

    parameters = {'kernel':('linear', 'rbf'), 'C':[0.00001,1]}
    svc = SVC(gamma="scale")
    clf = RandomizedSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)   
      

if __name__ == "__main__":

    X, y = load_data(".")
    F = extract_lbp_features(X)
    print("X shape: " + str(X.shape))
    print("F shape: " + str(F.shape))

    best_score, C_result, penalty_final = paramenter_searcher_manual(F,y)
    #parameter_searcher_grid(F,y)
    #parameter_searcher_grid_rand(F,y)

