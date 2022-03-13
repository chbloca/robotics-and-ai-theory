# Copyright (C) 2018 Santiago Cortes, Juha Ylioinas
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.



# Preparations
from matplotlib.pyplot import imread
import numpy as np
from numpy.fft import fftshift, fft2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from utils import affinefit
import warnings
warnings.filterwarnings('ignore')


## Load test images
man = imread('man.jpg') / 255.
wolf = imread('wolf.jpg') / 255.

# the pixel coordinates of eyes and chin have been manually found 
# from both images in order to enable affine alignment 
man_eyes_chin=np.array([[502, 465], [714, 485], [594, 875]])
wolf_eyes_chin=np.array([[851, 919], [1159, 947], [975, 1451]])
A, b = affinefit(man_eyes_chin, wolf_eyes_chin)

xv, yv = np.meshgrid(np.arange(0, man.shape[1]), np.arange(0, man.shape[0]))
pt = np.dot(A, np.vstack([xv.flatten(), yv.flatten()])) + np.tile(b, (xv.size,1)).T
wolft = map_coordinates(wolf, (pt[1,:].reshape(man.shape), pt[0,:].reshape(man.shape)))

## Below we simply blend the aligned images using additive superimposition
additive_superimposition = man + wolft

## Next we create two different Gaussian kernels for low-pass filtering
## the two images


# naive blending by additive superimposition for illustration
superimpose = man + wolft

# low-pass filter the two images using two different Gaussian kernels
sigmaA = 16
sigmaB = 8
man_lowpass = gaussian_filter(man, sigmaA, mode='nearest')
wolft_lowpass = gaussian_filter(wolft, sigmaB, mode='nearest')
# We use gaussian_filter in this case as it is significantly faster


## Your task is to create a hybrid image by combining a low-pass filtered 
## version of the human face with a high-pass filtered wolf face
 
## HINT: You get a high-pass version by subtracting the low-pass filtered version
## from the original image. Experiment also by trying different values for
## 'sigmaA' and 'sigmaB' above.
 
## Thus, your task is to replace the zero image on the following line
## with a high-pass filtered version of 'wolft'

wolft_highpass = np.zeros(man_lowpass.shape);

##--your-code-starts-here--##

##--your-code-ends-here--##
 
## Replace also the zero image below with the correct hybrid image
hybrid_image = np.zeros(man_lowpass.shape)

##--your-code-starts-here--##

##--your-code-ends-here--##
 

## Notice how strongly the interpretation of the hybrid image is affected 
## by the viewing distance

## Display input images and both output images.
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
plt.suptitle("Results of superimposition", fontsize=20)
ax = axes.ravel()
ax[0].imshow(man, cmap='gray')
ax[0].set_title("Input Image A")
ax[1].imshow(wolft, cmap='gray')
ax[1].set_title("Input Image B")
ax[2].imshow(additive_superimposition, cmap='gray')
ax[2].set_title("Additive Superimposition")
ax[3].imshow(hybrid_image, cmap='gray')
ax[3].set_title("Hybrid Image")
plt.subplots_adjust(top=1.2)
plt.show()

## Finally, visualize the log magnitudes of the Fourier
## transforms of the original images. 
## Your task is to calculate 2D fourier transform 
## for wolf/man and their filtered results using fft2 and fftshift

##--your-code-starts-here--##

##--your-code-ends-here--##

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
plt.suptitle("Magnitudes of the Fourier transforms", fontsize=20)
ax = axes.ravel()

ax[0].imshow(np.log(np.abs(F_man)), cmap='gray')
ax[0].set_title("log(abs(F_man))")
ax[1].imshow(np.log(np.abs(F_man_lowpass)), cmap='gray')
ax[1].set_title("log(abs(F_man_lowpass)) image")
ax[2].imshow(np.log(np.abs(F_wolft)), cmap='gray')
ax[2].set_title("log(abs(F_wolft)) image")
ax[3].imshow(np.log(np.abs(F_wolft_highpass)), cmap='gray')
ax[3].set_title("log(abs(F_wolft_highpass))")
plt.subplots_adjust(top=1.2)
plt.show()