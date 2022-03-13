import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def SignalGen(f0):
    #3a Sin
    zeros_1 = np.zeros(500)
    zeros_2 = np.zeros(300)
    n = np.arange(100)
    x = np.concatenate((zeros_1, n, zeros_2), axis=0)
    y = np.sin(2 * np.pi * f0 * x)
    plt.figure(1)
    plt.subplot(411)
    plt.title("Raw Signal")
    plt.plot(y)
    #3b Noise+ Sin
    w = np.sqrt(0.5) * np.random.randn(y.size)
    y_n = y + w
    plt.subplot(412)
    plt.title("Signal + Noise")
    plt.plot(y_n)
    #3c Detectors
    #Method 1
    res = []
    wsz = 100
    zeros_3 = np.zeros(100)
    limit = len(y_n)-wsz
    for w in range(limit):
        wndw = y_n[w:w+wsz-1]
        y = np.convolve(np.cos(2*np.pi*f0*n), wndw, 'same')
        res = np.append(res, y)
    res = np.concatenate((res, zeros_3), axis=0)
    plt.subplot(413)
    plt.title("Method 1 - Known signal")
    plt.plot(res)
    #Method 2
    h = np.exp(-2 * np.pi * 1j * f0 * n)
    res2 = np.abs(np.convolve(h, y_n, 'same'))
    plt.subplot(414)
    plt.title("Method 2 - Random signal")
    plt.plot(res2)
    plt.show()
def Classifiers():
    #5a Load data
    mat = loadmat("C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #3/twoClassData.mat")
    X = mat["X"]
    y = mat["y"].ravel()
    #5b random sample for training and test (200 each)
    a = shuffle(y)
    x0 = X[a == 0, :]
    x1 = X[a == 1, :]
    ytrain = y[a == 0]
    ytest = y[a == 1]
    plt.plot(x1[:, 0], x1[:, 1], 'bo')
    plt.plot(x0[:, 0], x0[:, 1], 'ro')
    plt.show()
    #5c KNN classifier
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    model.fit(x0, ytrain)
    resa = model.predict(x1)
    scorea = accuracy_score(ytest, resa)
    print('KNN score: ', scorea)
    #5d LDA classifier
    clf = LinearDiscriminantAnalysis()
    clf.fit(x0, ytrain)
    resb = clf.predict(x1)
    scoreb = accuracy_score(ytest, resb)
    print('LDA score: ', scoreb)
if __name__ == "__main__":
    SignalGen(0.1)
    #4 change frequency to 0.03
    SignalGen(0.03)
    #5 classifiers
    Classifiers()

