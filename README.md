# Dlib_detect_visualize_store-BuildDataset
Goal: to extract left_eye, right_eye, face, face_grid and label (train_y) into npy (npz) file. 


Goal: to extract left_eye, right_eye, face, face_grid and label (train_y) into npy (npz) file. 

1) load thousands of images from one file path
 
	example file name: filename.id.ss.imageframe.label.jpg

	label = 0 or 1.jpg

	0= good pose
	1= negative pose

2) call dlib predictor for extraction

3) loop to extract left_eye, right_eye, face, face_grid for thousands of frame, if no face and eye detected, drop frame. At the end, calculate how many frame has extracted face and eye. 

4) visualize random 10 example frames of extracted left_eye, right_eye, face, face_grid

5) store extracted left_eye, right_eye, face, face_grid as npy 

	left_eye, right_eye, face, face_grid --> npy

	filename.id.ss.imageframe.label.jpg --> npy


6) save 5 npy file: 

	left_eye.npy 
	right_eye.npy 
	face.npy
	face_grid.npy
	train_y.npy


7) load these five npy file to (test)

filename : itracker_adv.py

https://github.com/hugochan/Eye-Tracker 
