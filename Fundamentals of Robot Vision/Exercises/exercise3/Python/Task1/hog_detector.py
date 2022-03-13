import cv2
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.rcParams['figure.dpi']= 180


def inside(r, q):
    """
    Returns true if rectangle r = (x,y,w,h) is inside rectangle q.
    """
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


# Initalize HOG people detector
hog = cv2.HOGDescriptor()
detectorCoefficients = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(detectorCoefficients)  # we're using a support vector machine to classify detections

# Load test images
filename = 'people.jpg'    
img = cv2.imread(filename)


# Your task is to find suitable HOG detector parameters so that you're able to 
# get at least a few true positives. The most important parameters are 
# winSride and scale.
#
# The parameters are as follows:
#   winStride: A 2-tuple that is the “step size” in both the x and y 
#              direction of the sliding window.
#
#   scale: Controls the factor in which our image is resized at each layer of
#          the Gaussian image pyramid, ultimately influencing the number of levels
#          in the image pyramid. A smaller scale will increase the number of 
#          layers in the image pyramid, but also the processing time.
#
#   padding:  A tuple which indicates the number of pixels in both the x and y 
#             direction in which the sliding window ROI is “padded” prior to
#             HOG feature extraction.
#
#   hitThreshold: Threshold for the distance between features and Support Vector Machine (SVM)
#                 classifying plane. This can be set to a value above 0 if there
#                 is a large amount of false positives.
#
# Start by finding a value for scale (between 1.0-2.0, higher values are more 
# computationally efficient) which yields some sort of results. Next, try 
# decreasing winStride to achieve more detections. Finally, try increasing 
# hitTreshold to get rid of false positives. After this you can try to optimize
# the parameters even more by simply trying out different values. Pay also attention
# to the execution time.


##--your-code-start-here--##
scale = 1.07
winStride = (7,7)
padding = (32,32)
hitThreshold = 0.2
##--your-code-ends-here--##


# Detect
start = time.time()
found, w = hog.detectMultiScale(img, 
                                winStride=winStride, 
                                padding=padding, 
                                scale=scale,
                                hitThreshold = hitThreshold)
print('Detector execution time: {:.2f} s'.format(time.time() - start))


# Filter overlapping detections
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and inside(r, q):
            break
    else:
        found_filtered.append(r)

print('(%d) (%d) persons found' % (len(found_filtered), len(found)))



# Draw the filtered detections
for x, y, w, h in found_filtered:
    cv2.rectangle(img, (x, y),
                  (x+w, y+h),
                  (0, 255, 0), thickness=5)

plt.imshow(img[..., ::-1])
plt.show()

