# Face Recognition with Edge TPU

This repository contains a trained model for the Coral edge TPU, creating embedding representations for
facial recognition. This allows fast inference on a local device. Face recognition is a two stage task: First faces are detected 
(resulting in a cropped image with the face) and in the second stage these images are transformed to an
embedding representing that face (1-D array). For face detection the following pretrained model was used:
https://coral.ai/examples/detect-image/#run-the-example-for-face-detection. The model for the embeddings
takes as input 96*96 arrays, values between 0 and 1, and outputs a 1-D array of size 192. The Facenet paper 
(https://arxiv.org/abs/1503.03832) recommends an output size of 128, but this results in an error when converting
to an edge TPU model. As a base model MobileNetV2 is used. So far the model has been trained on a quater of
the VGGFace2 dataset. Updated models will follow.

In the following I will explain how this model can be used on a project with the Raspberry Pi to recognize
previously scanned people and give live audio feedback who is visible during a video stream.

## How to scan and recognize people
hardware:

Raspberry 4B 4Gb

Coral USB Accelerator

Raspberry Pi Camera v2

Mini External USB 2.0 Speaker

Before you continue, check if all libraries from requirements.txt are installed.

### Scan People
Run scan_people.py to use the face detector and save pictures of a single person to a folder during a video stream. 
Read more detailed instructions within the code file.

### Create Embeddings
Run create_embeddings.py to use the face embedding model, converting the extracted facial pictures to embedding
 arrays. 

### Update Labels
Give names to the scanned people. Each scanned person is associated with the folder number in the folder "scanned_people".
Open the file people_labels.txt to write in the left column the folder number and in the right column the name of the person.
Start by replacing "Alfred Maier" with the person scanned and saved in folder number "1".

### Run Face Recognition
Run recognize_face.py to detect faces in a live stream, create embeddings of detected people and compare
with the embeddings that have already been recored and labeled. Read more detailed instructions within the code file.


## Use without Edge TPU
The python scripts also work without Edge TPU. In each file there is a variable "ifEdgeTPU_1_else_0" that can be
set to 0.
