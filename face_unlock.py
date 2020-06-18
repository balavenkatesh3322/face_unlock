# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
#import _pickle as cPickle
import pickle
import face_recognition

import os
from PIL import Image, ImageDraw

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

with open("trained_knn_model.clf", 'rb') as f:
 	knn_clf = pickle.load(f)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])


while True:	

	# grab the current frame
	(grabbed, image1) = camera.read()
	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
	
	image = image1[:, :, ::-1]

	X_face_locations = face_recognition.face_locations(image)

	# If no faces are found in the image, return an empty result.
	if len(X_face_locations) != 0:
		
		# Find encodings for faces in the test iamge
		faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

		# Use the KNN model to find the best matches for the test face
		print(np.array(faces_encodings).shape)
		closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
		are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]

		#draw = ImageDraw.Draw(image)

		# Predict classes and remove classifications that aren't within the threshold
		predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
		for name, (top, right, bottom, left) in predictions:
			# Draw a box around the face using the Pillow module
			#draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
			if name not in "unknown":
				os.popen('gnome-screensaver-command -d && xdotool key Return')

			# There's a bug in Pillow where it blows up with non-UTF-8 text
			# when using the default bitmap font
			#name = name.encode("UTF-8")
			#rect = (left,top,right,bottom)
			#(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(image1, (left,bottom),(right,top), (0, 255, 0), 2)

			# show the face number
			cv2.putText(image1, "{}".format(name), (left-10, top-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			# Draw a label with a name below the face
			#text_width, text_height = draw.textsize(name)
			#draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
			#draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
	else:
		os.popen('gnome-screensaver-command -a')
	#print(predictions)
	#cv2.imshow("output", image)	
	cv2.imshow("output image",image1)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# clean up the camera and close any open windows
camera.release()
cv2.destroyAllWindows()