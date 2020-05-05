# Face Recognition with Edge TPU

This repository contains a trained model for the Coral edge TPU, creating embedding representations for
facial recognition. Face recognition is a two stage task: First faces are detected 
(resulting in a cropped image with the face) and in the second stage these images are transformed to an
embedding representing that face (1-D array). For face detection the following pretrained model was used:
(https://coral.ai/examples/detect-image/#run-the-example-for-face-detection). The model for the embeddings
takes as input 96*96 arrays, values between 0 and 1, and outputs a 1-D array of size 192. The Facenet paper 
(https://arxiv.org/abs/1503.03832) recommends an output size of 128, but this results in an error when converting
to an edge TPU model. As a base model MobileNetV2 is used. So far the model has been trained on a quater of
the VGGFace2 dataset. Updated models will follow.

In the following I will explain how this model can be used on a project with the Raspberry Pi to recognize
previously scanned people and give live audio feedback who is visible during a video stream.

## How to scan and recognize people

### Scan People
blabla

### Create Embeddings

### Update Labels

### Run Face Recognition
