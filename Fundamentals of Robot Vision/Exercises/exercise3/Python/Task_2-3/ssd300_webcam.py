from pathlib import Path
import time
from keras import backend as K
import numpy as np
from models.keras_ssd300 import ssd_300
import cv2
from scipy.misc import imresize



############################## MODEL ##########################################
# Copyright 2018 Pierluigi Ferrari (SSD keras port)

# Set the image size.
img_height = 300
img_width = 300

# Build the Keras model

K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=10,
                nms_max_output_size=400)

# Load the trained weights into the model.
weights_path = Path('weights/VGG_VOC0712_SSD_300x300_iter_120000.h5')
model.load_weights(weights_path, by_name=True)

# Classes
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'doggo',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

###############################################################################




##################################### TASK ####################################
#
# NOTE: The video player can be stopped by selecting the video player window
#       and pressing 'q'
#
# Your task: 
#   1. Read a frame from camera using cam.read
#    
#   2. Mirror the frame using cv2.flip
#
#   3. Make a resized copy of the frame for the network.
#      Use the imported function imresize and
#      size (img_height, img_width) (defined above).
#
#   4. You need an additional first dimension (batch_size, now 1)
#      to feed the image to the network. This can be achieved with:
#      image = np.array([image])
#
#   5. Use this image to get predictions from the network. 
#      This can be done with preds = model.predict(input)
#      Predictions are in the following form: 
#      [class_id, confidence, xmin, ymin, xmax, ymax]
#      These predictions should also be thresholded according to 
#      the prediction confidence. Anything above 0
#      (i.e omit 0 confidence values) should be used as a valid prediction.
#
#   6. The bounding box coordinates are for the resized image 
#      (size: img_height, img_width). Transform these coordinates to the original
#      frame by multiplying x coordinates with the ratio of original and reshaped widths, 
#      and y coordinates with the ratio of heights. 
#      Remember to also convert these to integer-type.
#
#   7. Use cv2.rectangle to insert a bounding box to the original 
#      (non-resized) frame.
#
#   8. Use cv2.putText to insert the predicted class label to the bounding box.
#      Class labels are defined above, and can be indexed with the class_id
#      in the predictions.
# 
###############################################################################


# Create a camera instance
cam = cv2.VideoCapture(0)

# Check if instantiation was successful
if not cam.isOpened():
    raise Exception("Could not open camera")


while True:    
    start = time.time()  # timer for FPS
    img = np.zeros((480, 640, 3))  # dummy image for testing
    
    #######-your-code-starts-here-########
    #1
    ret, frame = cam.read()
    
    #2
    frame = cv2.flip(frame, 1) # horizontal flipping (y axis)
    
    #3
    resized_frame = imresize(frame, [img_width, img_height])  # resize the image to the tuple input
    
    #4
    resized_frame = np.array([resized_frame]) # image converted to an array
    
    #5
    preds = model.predict(resized_frame)
    preds_classes = preds[0]
    
    #6
    
    for box in preds_classes:
        if box[1] > 0: # if confidence is greater than 0
            xmin_orig = int(box[2] * img.shape[0] / img_width)
            ymin_orig = int(box[3] * img.shape[1] / img_height)
            xmax_orig = int(box[4] * img.shape[0] / img_width)
            ymax_orig = int(box[5] * img.shape[1] / img_height)
        
    #7 cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #do for
        cv2.rectangle(img, (xmin_orig,ymin_orig), (xmax_orig, ymax_orig), (0, 255, 0), 1)
    
    #8
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,classes[int(box[0])],(10,500), font, 4,(255,255,255),2)

    #######-your-code-ends-here-########     
    
    # Insert FPS/quit text and show image
    fps = "{:.0f} FPS".format(1/(time.time()-start))
    img = cv2.putText(img, fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                      fontScale=0.8, color=(0,255,255))
    img = cv2.putText(img, 'Press q to quit', (440, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=(0,0,255))
    cv2.imshow('Video feed', img)    
    
    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()   
cv2.destroyAllWindows()