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


# Continue your code here...
# Task 4

def KNN(F, y):

    model_KNN = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    score_KNN = cross_val_score(model_KNN, F, y, cv=cvn)
    print('KNN Results: ', score_KNN)
    return score_KNN

def LDA(F, y):

    model_LDA = LinearDiscriminantAnalysis()
    score_LDA = cross_val_score(model_LDA, F, y, cv=cvn)
    print('LDA Results: ', score_LDA)
    return score_LDA

def SVC_model(F, y):

    model_SVC = SVC(gamma='auto')
    score_SVC = cross_val_score(model_SVC, F, y, cv=cvn)
    print('SVC Results: ', score_SVC)
    return score_SVC

def LR(F, y):

    model_LR = LogisticRegression()
    score_LR = cross_val_score(model_LR, F, y, cv=cvn)
    print('LR Results: ', score_LR)
    return score_LR

# Task 5

#Random_Forest
def RF(F, y):

    model_RF = RandomForestClassifier()
    score_RF = cross_val_score(model_RF, F, y, cv=cvn)
    print("RF Results: ", score_RF)

#Extr_Rand_Trees
def ERT(F, y):

    model_ERT = ExtraTreesClassifier(n_estimators=100)
    score_ERT = cross_val_score(model_ERT, F, y, cv=cvn)
    print("ERT Results: ", score_ERT)

#AdaBoost
def AB(F, y):

    model_AB = AdaBoostClassifier()
    score_AB = cross_val_score(model_AB, F, y, cv=cvn)
    print("AB Results: ", score_AB)

#GradientBoost
def GBT(F, y):

    model_GBT = GradientBoostingClassifier()
    score_GBT = cross_val_score(model_GBT, F, y, cv=cvn)
    print("GBT Results: ", score_GBT)

if __name__ == "__main__":

    X, y = load_data(".")
    F = extract_lbp_features(X)
    print("X shape: " + str(X.shape))
    print("F shape: " + str(F.shape))
    KNN(F, y)
    LDA(F, y)
    SVC_model(F, y)
    LR(F, y)
    RF(F, y)
    ERT(F, y)
    AB(F, y)
    GBT(F, y)





