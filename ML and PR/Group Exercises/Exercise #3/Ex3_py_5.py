from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def KNN(X_train, X_test, y):
    model = KNeighborsClassifier(n_neighbors = 2, metric = "euclidean")
    Y = model.fit(X_test, y[200:])
    acc = accuracy_score(X_test, Y)
    return Y
    
    
def LDA():
    clf = LinearDiscriminantAnalisys()
    clf.fit(X_train, y[200:])
    acc = accuracy_score(X_test, Y)


if __name__ == "__main__":
    mat = loadmat("twoClassData.mat")
    print(mat.keys())
    X_train = mat["X"][:200]
    X_test = mat["X"][200:]
    y = mat["y"].ravel() #or y.ravel(), ravel(y)
    
    #print(X[y==0,:])
    #plt.plot(X[0:200,0], X[0:200,1], 'bo')
    #plt.plot(X[200:400,0], X[200:400,1], 'ro')
    
    KNN(X_train, X_test, y)
    LDA(X_train, X_test, y)