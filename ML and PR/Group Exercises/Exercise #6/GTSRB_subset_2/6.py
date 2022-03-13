# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:48:51 2019

@author: hehu
"""

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

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


X, y = load_data(".")
X_norm = normalization(X)
y = np.vstack((y, np.abs(1-y))).T
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.2)

N = 32
num_classes = 2

model = Sequential()

model.add(Conv2D(N, (5, 5), input_shape=(64, 64, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(N, (5, 5), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Flatten())
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(num_classes, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data = ([X_test, y_test]))




