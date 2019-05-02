# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import os
import argparse
import dlib
from collections import OrderedDict
from imutils import face_utils
import numpy as np
import cv2
import imutils
import math
import glob
from PIL import Image

# load thousands of images from one file path
# load all images from path
TrainCroppedLabelledImage_inputPath = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped'
TrainCroppedLabelledImage = glob.glob(os.path.join(TrainCroppedLabelledImage_inputPath, '*.jpg')) #sorted
TrainCroppedLabelledImage_list = []

# imagefilename = /home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped/Cropped.01.s01.KinectFrame5000.0.jpg

right_eye = []
left_eye = []
face = []

# right_eye_data = []
# left_eye_data = []
# face_data = []

# train_y_data = []

roi_frame = []

width = 64
height = 64


# call dlib predictor for detection and extraction.
predictor = dlib.shape_predictor(
    '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/dlib-models-master/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
print('initialize detector')

# need to adjust the points and call from this point
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("right_eye", (37, 42)),
    ("left_eye", (42, 48)),
    ("face", (1, 68)),
])

n_images = 5
n_channels = 3
all_left_eyes = np.zeros(shape=(n_images, width, height, n_channels), dtype=np.uint8)   #shape (8000, 64, 64, 3)
all_right_eyes = np.zeros(shape=(n_images, width, height, n_channels), dtype=np.uint8)  #shape (8000, 64, 64, 3)
all_face = np.zeros(shape=(n_images, width, height, n_channels), dtype=np.uint8)        #shape (8000, 64, 64, 3)

all_label = np.zeros(shape=(n_images, 1), dtype=np.float64)  #shape (8000, 1)

# define general rules

# count = false, while true, detect face and eye (must be three detection :face, left eye, right eye)
# and save as numpy. and then, convert file as label


# Rule = False
count = 0
totalfile = 0

def get_label(img_name):
    label = float(img_name.split('.')[-2])
    print ('Label is ', label)
    print (img_name)
    return label


for n, img in enumerate(TrainCroppedLabelledImage[:n_images]):
    totalfile = totalfile + 1
    image = cv2.imread(img)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # insert into all_label
    label = get_label(img)
    all_label[n] = label

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over face detection
    for (r, rect) in enumerate(rects):
        # determine the facial landmarks for the face region
        # convert the landmark (x,y)-coordinates to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        detect = False

        # loop over an individual facial parts
        for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
            visual_check = image.copy()

            # drop the image_file if all the three parts (face/left eye or right eye) is not detected
            # if all face parts detected, save and convert filename to label
            # only extract and save if face and both eye detected. if only either one detected, then drop the file
            # if (name == "face", "right_eye", "left_eye"):
            count = count +1
            detect = True

            # draw rectangle on detected facial parts
            (x, y, w, h) = cv2.boundingRect (np.array([shape[i:j]]))
            cv2.rectangle(visual_check, (x, y), (x + w, y + h), (0, 255, 0), 2)     #green box

            # clone the original image
            # display the name of the face part on the image
            extracted_data = image.copy()
            cv2.putText(visual_check, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

            # loop over the subset of facial landmarks, drawing the specific face parts
            for (x, y) in shape[i:j]:
                cv2.circle(visual_check , (x, y), 1, (0, 0, 255), -1) #red facial mark

                #extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                offset = abs(w-h)/2
                extraspace = 25
                if w > h:
                    roi = extracted_data[y - math.floor(offset) - extraspace: y + h + math.ceil(offset) + extraspace,
                          x - extraspace: x + w + extraspace]
                else:
                    roi = extracted_data[y - extraspace: y + h + extraspace,
                          x - math.floor(offset) - extraspace: x + w + math.ceil(offset) + extraspace]
                roi_frame = imutils.resize(roi, width=height, inter=cv2.INTER_CUBIC)

            if (name == "right_eye"):
                all_right_eyes[n] = roi_frame

            elif (name == "left_eye"):
                all_left_eyes[n] = roi_frame

            elif (name == "face"):
                all_face[n] = roi_frame

            if n == 2:
                # show the particular face part (comment to skip)
                cv2.imshow("ROI " + img.split('/')[-1] + ' ' + str(all_label[n]), roi_frame)
                cv2.imshow("Frame" + img.split('/')[-1] + ' ' + str(all_label[n]), visual_check)   #how to display filename with label for check ?
                cv2.waitKey(0)

            if detect:
                np.save('./train_right_eye.npy', all_right_eyes[:,:,:,::-1])
                np.save('./train_left_eye.npy', all_left_eyes[:,:,:,::-1])
                np.save('./train_face.npy', all_face[:,:,:,::-1])

        print('done extracted', (roi_frame.shape))
        print('Total_Face+RightEye+LeftEye=', count)
    print("Total_File_Pass=", totalfile)
print("Total_File_with_FaceAndEyes_Detected",(count/3))

np.save('./train_label_y.npy', all_label)























                # print("number of True file (face and eye detected", len(rects) )


                # count final detected file which consist face, right eye and left eye
















###############################################################################################################




# store extracted left_eye, right_eye, face as npy
#
# 	left_eye, right_eye, face --> npy
#
# 	filename.id.ss.imageframe.label.jpg --> npy

            # 	left_eye.npy        (1000000, 64, 64, 3)
            # 	right_eye.npy       (1000000, 64, 64, 3)
            # 	face.npy            (1000000, 64, 64, 3)

            # 	train_y.npy         (1000000, 1)



# load these five npy file to (test)
#
# filename : itracker_adv.py
#
# https://github.com/hugochan/Eye-Tracker