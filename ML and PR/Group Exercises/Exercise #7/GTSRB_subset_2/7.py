# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:48:51 2019

@author: hehu
"""

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.applications.vgg16 import VGG16
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_mat():
    #4. RFECV
    mat = sp.loadmat('arcene.mat')
    xtest=mat["X_test"]
    xtrain = mat["X_train"]
    ytrain = mat["y_train"].ravel()
    ytest = mat["y_test"].ravel()
    return xtest, xtrain, ytest, ytrain
def rfe_sel(xtest, xtrain, ytest, ytrain):
    rfe = RFECV(estimator=LinearDiscriminantAnalysis(), step=50)#verbose=1
    rfe.fit(xtrain, ytrain)
    plt.plot(range(0,10001,50),rfe.grid_scores_)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
    resrfe = rfe.predict(xtest)
    scorerfe = accuracy_score(ytest, resrfe)
    return scorerfe, np.sum(rfe.support_)
def l1_sel(xtest, xtrain, ytest, ytrain):
    clf= LogisticRegression(penalty='l1')
    C_range = 10.0 ** np.arange(-5,6,0.5)
    accuracies = []
    nonzeros = []
    bestScore = 0
    for C in C_range:
        clf.C = C
        score = cross_val_score(clf, xtrain, ytrain, cv=10).mean()
        accuracies.append(score)
    bestScore=max(accuracies)
    bestc = C_range[np.argmax(accuracies)]
    clf.C = bestc
    clf.fit(xtrain,ytrain)
    resclf = clf.predict(xtest)
    scoreres = accuracy_score(ytest, resclf)
    return bestScore, bestc, scoreres, np.count_nonzero(clf.coef_)
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

def normalization(X):
    numerator = X - np.min(X)
    denominator = np.max(X)
    X_norm = np.divide(numerator, denominator)
    return X_norm

def vgg16net ():
    X, y = load_data(".")
    X_norm = normalization(X)
    y = np.vstack((y, np.abs(1 - y))).T
    # or alternatively y_binary = to_categorical(y_int)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
    w = base_model.output
    w = Flatten()(w)
    w = Dense(100, activation='relu')(w)
    output = Dense(2, activation='sigmoid')(w)
    model = Model(inputs=[base_model.input], outputs=[output])
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True
    model.layers[-7].trainable = True
    model.summary()
    model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data = ([X_test, y_test]))


if __name__ == "__main__":
    vgg16net()
    xtest, xtrain, ytest, ytrain = load_mat()
    scorerfe, featrfe = rfe_sel(xtest, xtrain, ytest, ytrain)
    l1score, l1c, l1testsc, l1nfeatures = l1_sel(xtest, xtrain, ytest, ytrain)
    print("Number of features selected on RFE: ", featrfe, "\nRFE accuracy: ", scorerfe, )
    print("Best score: ", l1score, " for C: ", l1c,"\n Test accuracy:",l1testsc," Number of features: ", l1nfeatures)
