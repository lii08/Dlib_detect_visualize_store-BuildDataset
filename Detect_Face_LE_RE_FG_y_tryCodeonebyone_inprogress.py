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
import glob

# 1) call dlib predictor for detection and extraction.
predictor = dlib.shape_predictor(
    '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/dlib-models-master/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
print('initialize detector')

# need to adjust the points and call from this point
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("right_eye", (37, 42)),
    ("left_eye", (42, 48)),
    ("face", (1, 68)),
    ("face_grid", (49, 68)),
])

# 2) load thousands of images from one file path
# load all images from path
TrainCroppedLabelledImage_inputPath = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped'
TrainCroppedLabelledImage = glob.glob(os.path.join(TrainCroppedLabelledImage_inputPath, '*.jpg'))
TrainCroppedLabelledImage_list = []

right_eye = []
left_eye = []
face = []
face_grid = []

right_eye_data = []
left_eye_data = []
face_data = []
face_grid_data = []
train_y = []

width = height = 64


# load all images in traincroppedlabelledimage_inputPath
for n, img in enumerate(TrainCroppedLabelledImage[:5]):
    image = cv2.imread(img)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over face detection
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region
        # convert the landmark (x,y)-coordinates to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # draw box on detected face_grid
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # loop over an individual facial parts
        for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():

            # draw rectangle on detected facial parts
            (x1, y1, w1, h1) = cv2.boundingRect(np.array([shape[i:j]]))
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            # np.save.

            # clone the original image
            # display the name of the face part on the image
            # cv2 draw rect
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

            # loop over the subset of facial landmarks, drawing the specific face parts
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)


        right_eye_data.append(right_eye)
        left_eye_data.append(left_eye)
        face_data.append(face)
        face_grid_data.append(face_grid)

        # np.save('.../%train_label.npy', train_y)



        # show the particular face part
        cv2.imshow("Image", clone)
        cv2.waitKey(0)



print(right_eye_data)
print(left_eye_data)
print(face_grid_data)
print(face_data)

np.save('./train_right_eye.npy', right_eye_data)
np.save('./train_left_eye.npy', left_eye_data)
np.save('./train_face.npy', face_data)
np.save('./train_face_grid', face_grid_data)
















# # visualize all facial landmarks with a transparent overlay
# output = face_utilsvisualize_facial_landmarks(image, shape)
# cv2.imshow("Image", output)
# cv2.waitKey(0)

# 3) loop to extract left_eye, right_eye, face, face_grid for thousands of frame, if no face and eye detected, drop frame. \\
# At the end, calculate how many frame has extracted face and eye. Draw rectangle box for detected eye and face


# 4) visualize random 10 example frames of extracted left_eye, right_eye, face, face_grid


# 5) store extracted left_eye, right_eye, face, face_grid as npy
#
# 	left_eye, right_eye, face, face_grid --> npy
#
# 	filename.id.ss.imageframe.label.jpg --> npy


# for f in os.listdir(input_path):
#     if f.find(".png") != -1:
#         img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
#         file_name = f[:f.find(".png")]
#
#         np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
#         retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]
#
#         assert np.array_equal(img, retrieve)
#
#         shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))
#
# print("Numpy arrays saved in {}".format(output_path))


# 6) save 5 npy file:
#
# 	left_eye.npy
# 	right_eye.npy
# 	face.npy
# 	face_grid.npy
# 	train_y.npy


# 7) load these five npy file to (test)
#
# filename : itracker_adv.py
#
# https://github.com/hugochan/Eye-Tracker