import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from PIL import Image
import cv2

# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/eye_tracker_train_and_val.npz_FILES/train_eye_left.npy'    #(48000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/eye_tracker_train_and_val.npz_FILES/train_eye_right.npy'  #(48000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/eye_tracker_train_and_val.npz_FILES/train_face.npy'       #(48000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/eye_tracker_train_and_val.npz_FILES/train_face_mask.npy'        #(25 x25)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/eye_tracker_train_and_val.npz_FILES/train_y.npy'           #(48000, 2)


file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/PreprocessPythonCode/train_face.npy'          #(8000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/PreprocessPythonCode/train_left_eye.npy'    #(8000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/PreprocessPythonCode/train_right_eye.npy'   #(8000, 64, 64, 3)
# file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/PreprocessPythonCode/train_label_y.npy'       #dtype = float64



# file_name = glob.glob(os.path.join(file_path, '*.npy'))


img_array = np.load(file_path)

print(img_array)
print(img_array.dtype)
print(img_array.ndim)
print(img_array.shape)
print(img_array.size)

for x in img_array[:5]:
    # x = x[:,:,::-1]
    # if np.all(x == 0):
    #     continue
    plt.imshow(x)
    plt.show()




