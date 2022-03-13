import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from trianglin import trianglin
matplotlib.rcParams['figure.dpi']= 180

# Visualization of three point correspondences in both images
im1 = plt.imread('im1.jpg')
im2 = plt.imread('im2.jpg')

# Points L, M, N in image 1
lmn1 = 1.0e+03 * np.array([[1.3715, 1.0775], 
                        [1.8675, 1.0575], 
                        [1.3835, 1.4415]])

# Points L, M, N in image 2
lmn2 = 1.0e+03 * np.array([[1.1555, 1.0335],
                        [1.6595, 1.0255],
                        [1.1755, 1.3975]])
                   
labels = ['L', 'M', 'N']                 
plt.figure(1)
plt.imshow(im1)
for i in range(len(labels)):    
    plt.plot(lmn1[i, 0], lmn1[i, 1], 'c+', markersize=10)
    plt.annotate(labels[i], (lmn1[i, 0], lmn1[i, 1]), color='c', fontsize=20)
plt.xticks([])
plt.yticks([])
plt.show()


plt.figure(2)
plt.imshow(im2)
for i in range(len(labels)):    
    plt.plot(lmn2[i, 0], lmn2[i, 1], 'c+', markersize=10)
    plt.annotate(labels[i], (lmn2[i, 0], lmn2[i, 1]), color='c', fontsize=20)
plt.xticks([])
plt.yticks([])
plt.show()

# The task is to implement the missing function in the file 'trianglin.py'.
# The algorithm is described in the exercise sheet.
# Output should be the homogeneous coordinates of the triangulated point.

# Load the projection matrices
P1 = np.load('P1.npy')
P2 = np.load('P2.npy')


# Triangulate
L = trianglin(P1, P2, np.hstack((lmn1[0,:].T, [1])), 
                      np.hstack((lmn2[0,:].T, [1])))
M = trianglin(P1, P2, np.hstack((lmn1[1,:].T, [1])), 
                      np.hstack((lmn2[1,:].T, [1])))
N = trianglin(P1, P2, np.hstack((lmn1[2,:].T, [1])), 
                      np.hstack((lmn2[2,:].T, [1])))
                      
# We can then use these world coordinates to compute the width and height 
# of the picture on the book
picture_w_mm = np.linalg.norm(L[0:3] / L[3] - M[0:3] / M[3])
picture_h_mm = np.linalg.norm(L[0:3] / L[3] - N[0:3] / N[3])
print("Picture width: %.2f mm" % picture_w_mm.item())
print("Picture height: %.2f mm" % picture_h_mm.item())
