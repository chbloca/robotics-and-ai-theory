import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from utils import vgg_X_fromxP_lin, camcalibDLT
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib
matplotlib.rcParams['figure.dpi']= 160

# Load the images and their camera matrixes
# (two views of the same scene)
im1 = np.array(Image.open('im1.jpg'))
im2 = np.array(Image.open('im2.jpg'))
P1 = loadmat('P1.mat')['P1']
P2 = loadmat('P2.mat')['P2']

# Give labels for the corners of the shelf
labels = ['a','b','c','d','e','f','g','h']

# Load image coordinates for the corners
x1 = loadmat('cornercoordinates.mat')['x1']
x2 = loadmat('cornercoordinates.mat')['x2']
y1 = loadmat('cornercoordinates.mat')['y1']
y2 = loadmat('cornercoordinates.mat')['y2']

# Define the 3D coordinates of the corners based on the known dimensions
ABCDEFGH_w = np.array([[758, 0, -295],
                    [0, 0, -295],
                    [758, 360, -295],
                    [0, 360, -295],
                    [758, 0 ,0],
                    [0, 0, 0],
                    [758, 360, 0],
                    [0, 360, 0]])
          
# Visualize the corners in the images
plt.figure(1)
plt.imshow(np.hstack((im1,im2)))
plt.xticks([])
plt.yticks([])
plt.plot(x1, y1, 'c+', markersize=10)
plt.plot(x2+np.shape(im1)[1], y2, 'c+', markersize=10)

for i in range(np.size(x1)):
    plt.annotate(labels[i], (x1[i], y1[i]), color='c', fontsize=20)
    plt.annotate(labels[i], (x2[i]+np.shape(im1)[1], y2[i]), color='c', fontsize=20)

# Calibrate the cameras from 3D <-> 2D correspondences
P1t = camcalibDLT(np.hstack((ABCDEFGH_w, np.ones((8,1)))),
                  np.hstack((x1, y1, np.ones((8,1)))))
P2t = camcalibDLT(np.hstack((ABCDEFGH_w, np.ones((8,1)))),
                  np.hstack((x2, y2, np.ones((8,1)))))
                  
# Visualize a 3D sketch of the shelf
edges = np.array([[0,1], [0,2], [2,3], [1,3], [0,4], [4,5], [1,5],
               [4,6], [2,6], [3,7], [6,7], [5,7]]).T
    
plt.figure(2)
ax = plt.axes(projection='3d')
plt.title('3D sketch of the shelf')
for i in range(np.shape(edges)[1]):
    ax.plot3D(ABCDEFGH_w[edges[:, i], 0], ABCDEFGH_w[edges[:, i], 1], ABCDEFGH_w[edges[:, i], 2], 'gray')
for i in range(8):
    ax.text(ABCDEFGH_w[i][0], ABCDEFGH_w[i][1], ABCDEFGH_w[i][2], labels[i], fontsize=20)

# Project the 3D corners to images
corners1 = np.dot(P1t, np.vstack((ABCDEFGH_w.T, np.ones(8))))
corners2 = np.dot(P2t, np.vstack((ABCDEFGH_w.T, np.ones(8))))
cx1 = (corners1[0,:] / corners1[2,:]).T
cy1 = (corners1[1,:] / corners1[2,:]).T
cx2 = (corners2[0,:] / corners2[2,:]).T
cy2 = (corners2[1,:] / corners2[2,:]).T

# Illustrate the edges of the shelf that connect its corners
plt.figure(1)
plt.axis('equal')
for i in range(np.shape(edges)[1]):
    plt.plot(cx1[edges[:,i]], cy1[edges[:,i]], 'm-')
    plt.plot(cx2[edges[:,i]]+np.shape(im1)[1], cy2[edges[:,i]], 'm-')
    
# Compute a projective reconstruction of the shelf
# That is, triangulate the corner correspondences using the camera projection
# matrices which were recovered from the fundamental matrix

Xcorners = np.zeros((4,8))
for i in range(8):
    # the following function is from http://www.robots.ox.ac.uk/~vgg/hzbook/code/
    imsize = np.array([[np.shape(im1)[1], np.shape(im2)[1]],[np.shape(im1)[0], np.shape(im2)[0]]])
    P = [P1, P2]
    u = np.array([[x1[i], x2[i]],[y1[i], y2[i]]])
    Xcorners[:,i] = vgg_X_fromxP_lin(u, P, imsize) 
Xc = Xcorners[0:3,:] / Xcorners[[3,3,3], :]

# Visualize the projection reconstruction
# Notice that the shape is not a rectangular cuboid
# (there is a projective distortion)
plt.figure(3)

ax = plt.axes(projection='3d')
plt.title('Projection reconstruction')
for i in range(np.shape(edges)[1]):
    ax.plot3D(Xc[0, edges[:, i]], Xc[1, edges[:, i]], Xc[2, edges[:, i]], 'gray')
for i in range(8):
    ax.text(Xc[0][i], Xc[1][i], Xc[2][i], labels[i], fontsize=20)


# Your task is to project the cuboid corners 'Xc' to images 1 and 2.
# Use camera projection matrices P1 and P2.
# Visualize the results using cyan lines.
# The cyan edges should be relatively close to the magenta lines which are alerady plotted.

##-your-code-starts-here-##

##-your-code-ends-here-##

# Uncomment these after you've calculated the projected points
# pcx1 and pcy1 are x and y coordinates for image 1, and similarly for image 2
#plt.figure(1)
#plt.title('Cyan: projected cuboid')
#for i in range(np.shape(edges)[1]):
#    plt.plot(pcx1[edges[:,i]], pcy1[edges[:,i]], 'c-')
#    plt.plot(pcx2[edges[:,i]]+np.shape(im1)[1], pcy2[edges[:,i]], 'c-')

