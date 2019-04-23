# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import os
import argparse
import dlib
# import the necessary packages
from collections import OrderedDict
from imutils import face_utils
import numpy as np
import cv2
import imutils
import glob

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())

# load image
TrainCroppedLabelledImage_inputPath = 'to_path'
TrainCroppedLabelledImage = glob.glob(os.path.join(TrainCroppedLabelledImage_inputPath, '*.jpg'))

TrainCroppedLabelledImage = []

image = []
shape = []
clone = []

right_eye = []
left_eye = []
face = []
face_grid = []
train_y = []

# Call predictor
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('_Path_to_Predictor')

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
    ("face", (1,68)),
    ("face_grid", (49,68)),
])

print ('check1')
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
for img in TrainCroppedLabelledImage:
    image = cv2.imread(TrainCroppedLabelledImage)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            print('name', name)
            # if name == LE,RE,....
            if name == "left_eye":
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif name == "right_eye":
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

            elif name == "face":
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)

            elif name == "face_grid":
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 0), 2)




                print ('check loop')
                print ("name", name)



                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                # clone = image.copy()
                # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 0, 255), 2)

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=224, inter=cv2.INTER_CUBIC)

                # show the particular face part
                # extract LE RE , ..



        # here save as npy for 5 output



# # visualize all facial landmarks with a transparent overlay
# output = face_utils.visualize_facial_landmarks(image, shape)
# cv2.imshow("Image", output)
# cv2.waitKey(0)


