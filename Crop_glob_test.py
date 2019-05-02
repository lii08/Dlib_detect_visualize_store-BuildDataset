import cv2
import os
import glob


Image_file_path = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/DATAdummyDell_RawLabelled/Test_dummyRawLabelled'


image_all = glob.glob(os.path.join(Image_file_path, '*.jpg'))

x = 240
y = 180
h = w = 224

for idx in image_all:
    img = cv2.imread(idx)
    crop_img = img[y:y+h, x:x+w]
    file_name = idx.split('/')[-1]

    cv2.imwrite(os.path.join('/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Test_dummy_cropped', 'Cropped.{}'.format(file_name)), crop_img)

# cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


print('All images was successfully cropped')