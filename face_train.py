import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
   
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory

    train_dir = os.listdir('dataset/')
    print(train_dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("dataset/" + person)
    
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            print("dataset/" + person + "/" + person_img)
            face = face_recognition.load_image_file("dataset/" + person + "/" + person_img)
            print(face.shape)
            # Assume the whole image is the location of the face
            #height, width, _ = face.shape
            # location is in css order - top, right, bottom, left
            height, width, _ = face.shape
            face_location = (0, width, height, 0)
            print(width,height)

            face_enc = face_recognition.face_encodings(face, known_face_locations=[face_location])
            
            

            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()
        
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

    print(np.array(encodings).shape)
    print(np.array(names).shape)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings,names)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf



if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("dataset", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")    