import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from utils import draw_eplines, vgg_F_from_P
from estimateF import estimateF
from estimateFnorm import  estimateFnorm 
import matplotlib
matplotlib.rcParams['figure.dpi']= 200

# Load images and their corresponding camera matrices
im1 = np.array(Image.open('im1.jpg'))
im2 = np.array(Image.open('im2.jpg'))
P1 = loadmat('P1.mat')['P1']
P2 = loadmat('P2.mat')['P2']


# The given image coordinates were originally localized manually.
# The point correspondences are the same as in round 7.
# That is, 11 points (A,B,C,D,E,F,G,H,L,M,N) are marked from both images.
labels = ['a','b','c','d','e','f','g','h','l','m','n']

x1 = 1.0e+03 * np.array([0.7435, 3.3315, 0.8275, 3.2835, 0.5475, 3.9875,
                      0.6715, 3.8835, 1.3715, 1.8675, 1.3835])
                      
y1 = 1.0e+03 * np.array([0.4455, 0.4335, 1.7215, 1.5615, 0.3895, 0.3895, 
                      2.1415, 1.8735, 1.0775, 1.0575, 1.4415])

x2 = 1.0e+03 * np.array([0.5835, 3.2515, 0.6515, 3.1995, 0.1275, 3.7475, 
                      0.2475, 3.6635, 1.1555, 1.6595, 1.1755])

y2 = 1.0e+03 * np.array([0.4135, 0.4015, 1.6655, 1.5975, 0.3215, 0.3135, 
                      2.0295, 1.9335, 1.0335, 1.0255, 1.3975])
                      
plt.figure(1)
plt.imshow(np.hstack((im1, im2)))

plt.plot(x1, y1, 'c+', markersize=5)
plt.plot(x2+np.shape(im1)[1], y2, 'c+', markersize=5)
for i in range(np.size(x1)):
    plt.annotate(labels[i], (x1[i], y1[i]), color='c', fontsize=10)
    plt.annotate(labels[i], (x2[i]+np.shape(im1)[1], y2[i]), color='c', fontsize=10)

# The fundamental matrix F can be computed from the projection matrices if they are known
FfromPs = vgg_F_from_P(P1, P2)

eplines = np.dot(FfromPs, np.vstack((x1, y1, np.ones(11))))
# Fx is the epipolar line associated with x, (l'= Fx)
# a = eplines[0,i] 
# b = eplines[1,i]
# c = eplines[2,i]
# ax+by+c=0
draw_eplines(eplines, im1, im2, 'c-')

# Implement the 8 point method and the normalized 8 point method for F-matrix estimation
# Uncomment the following lines after implementing 'estimateF.py' and 'estimateFnorm.py'

### The eight-point algorithm
#F = estimateF(np.vstack((x1, y1, np.ones(11))), np.vstack((x2, y2, np.ones(11))))
#
## The normalized eight-point algorithm
#Fnorm = estimateFnorm(np.vstack((x1, y1, np.ones(11))), np.vstack((x2, y2, np.ones(11))))
#
#eplinesA = np.dot(F, np.vstack((x1, y1, np.ones(11))))
#draw_eplines(eplinesA, im1, im2, 'm-')
#
#eplinesB = np.dot(Fnorm, np.vstack((x1, y1, np.ones(11))))
#draw_eplines(eplinesB, im1, im2, 'y-')
