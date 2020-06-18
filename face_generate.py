
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
import _pickle as cPickle

import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

number=0;
frame_count=0
detector = dlib.get_frontal_face_detector()
print("enter the person name")
name = input()
folder_name="dataset/"+name

if os.path.exists(folder_name):
    print ("Folder exist")
else:
	os.mkdir(folder_name)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])


while True:
	
	if frame_count % 5 == 0:

		print("keyframe")

		# grab the current frame
		(grabbed, image) = camera.read()
		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if args.get("video") and not grabbed:
			break
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# detect faces in the grayscale image
		rects = detector(gray, 1)
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then			
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			#print rect.dtype
			cro=image[y: y + h, x: x + w]

			out_image = cv2.resize(cro,(108,108))
			
			fram= os.path.join(folder_name+"/",str(number)+ "." + "jpg")
			number+=1		
			cv2.imwrite(fram,out_image)		
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)			
		frame_count+=1
		
	else:

		frame_count+=1
		print("redudant frame")

	if number >51:
		break			
	#cv2.imshow("output", image)	
	cv2.imshow("output image",image)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()