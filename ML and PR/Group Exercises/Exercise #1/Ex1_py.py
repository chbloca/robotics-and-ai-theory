import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D

locdata_P="C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #1/locationData/locationData.csv"
twoclass_P="C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #1/Ex1_data/twoClassData.mat"
lx_P="C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #1/least_squares_data/x.npy"
ly_P="C:/Users/chris/Google Drive/Documents/Studio/Master - 2nd Course (2018-2019)/ML and PR/Exercises/Exercise #1/least_squares_data/y.npy"

def read_a():
    locationData = np.loadtxt(locdata_P, dtype="float", delimiter=' ')
    print(locationData)
    shape_a = locationData.shape
    print(shape_a)
    return locationData, shape_a

def plot_a():
    fig_a = plt.figure()
    fig_a = plt.plot(locationData[:, 0], locationData[:, 1], label="plot_a")
    plt.title("plot_a")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.grid("on")
    plt.show()

def plot_b():
    ax = plt.subplot(1, 1, 1, projection = "3d")
    ax.plot(locationData[:, 0], locationData[:, 1], locationData[:, 2])
    plt.title("plot_b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.grid("on")
    plt.show()

def loops():
    X =[]
    values=0
    with open (locdata_P) as fp:
        for line in fp:
            values = line.split(" ")
            values = [float(v) for v in values]
            X.append(values)
    X=np.array(X)
    print(np.all([locationData,X]))

def load_mdata():
    mat = loadmat(twoclass_P)
    print(mat.keys())
    X = mat["X"]
    print(X.shape)
    y=mat["y"].ravel()
    print(y.shape)
    X0 = X[y == 0, :]
    plt.plot(X[:,0], X[:,1], 'bo')
    plt.plot(X0[:, 0], X0[:, 1], 'ro')
    plt.show()
    

def lsq():
    ls_x = np.load(lx_P)
    ls_y = np.load(ly_P)
    lstsq_x = np.vstack([ls_x, np.ones(len(ls_x))]).T
    a,b= np.linalg.lstsq(lstsq_x,ls_y)[0]
    print('A=',a," B=",b)
    plt.plot(ls_x, ls_y, 'o', label='Original data', markersize=10)
    plt.plot(ls_x, a * ls_x + b, 'r', label='Fitted line')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    #1 load csv file
    locationData, shape_a = read_a()
    #2.a Plot 2d
    plot_a()
    #2.b Plot 3d
    plot_b()
    #3 Use for loop
    loops()
    #4 Matlab data into Python
    load_mdata()
    #5 Least squares fit
    lsq()



