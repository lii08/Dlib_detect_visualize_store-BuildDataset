# test single file
import cv2

x = 240
y = 180
h = w = 224

img = cv2.imread("/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/DeepLearningExperimentDell/01.s01.KinectFrame5000.0.jpg")
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)



# labelled_raw_image_train_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/DATAdummyDell_RawLabelled/Train_dummyRawLabelled'
# image_name_dir = os.listdir('{}/'.format(labelled_raw_image_train_path))
#
#
# outputfile = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped'
#
# x1= 260
# x2 = 484
# y1 = 200
# y2 = 424
# (260, 484, 200, 424)
# 260:484, 200:424

from PIL import Image
import os.path, sys
import cv2
import matplotlib as plt

# imagePath = "/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/DATAdummyDell_RawLabelled/Train_dummyRawLabelled"
# image_path_list = os.listdir(imagePath)
#
# outputfile = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped'

# image_file = []

