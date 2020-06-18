# Lock/Unlock Ubuntu OS 

## Introduction
We can lock and unlock our Ubuntu system using face recognition(currently only on Ubuntu). 

## Requirements

Install below the required library in your local machine.

1) python 3.7
2) opencv 4.1.0
3) numpy 
4) face-recognition
5) sudo apt-get install gnome-screensaver
6) sudo apt-get install xdotool

## Quick Start
I have used three python files to solve this issue.

1) **face_generate.py**
 This will detect your face and save it in the dataset folder then the new folder will create with your name.
 
2) **face_train.py**
 This python file will open the dataset folder and take your image from that and train your face using the K-nearest neighbor algorithm and face_recognition library.
 
3) **face_unlock.py**
 This is an important python file that will detect your face using the webcam and unlock the system.


To do:
- [ ] Support Windows and Mac OS.
- [ ] Train face using Deep learning technique.
- [ ] Add feature to access at the UI level.