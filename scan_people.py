"""
Use this program to scan a singel person. Scanning means to use the face detector to extract a 96*96 img of the persons face.
The images are saved in one folder as png and in an other folder as numpy array. The png's are only for visual inspection.
For default values in "recognize_face.py" you need to get at least 20 pictures per person.

Important: For each new person, increase the variable "str(person_number)" in main() by 1. This "str(person_number)" will define the foldername
where the images of the scanned person will be saved.
"""

import io
import re
import time
import os
import shutil
from tflite_runtime.interpreter import load_delegate

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

import cv2

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960


def main():
  labels = load_labels('coco_labels.txt')
  interpreter = Interpreter(model_path = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  
  
  
  person_number = 1 # Change the number of the person you scan. It will create a new number for that person
  count_images_saved = 0
  
  if os.path.isdir('scanned_people/' + str(person_number)) == False:
    os.mkdir('scanned_people/' + str(person_number))
    os.mkdir('scanned_people/' + str(person_number) + '/png')
    os.mkdir('scanned_people/' + str(person_number) + '/npy')
  else:
    shutil.rmtree('scanned_people/' + str(person_number))
    os.mkdir('scanned_people/' + str(person_number))
    os.mkdir('scanned_people/' + str(person_number) + '/png')
    os.mkdir('scanned_people/' + str(person_number) + '/npy')
  

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
    camera.rotation=270
    camera.start_preview()
    try:
      stream = io.BytesIO()
      annotator = Annotator(camera)
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image_large = Image.open(stream)
        image = image_large.convert('RGB').resize(
            (input_width, input_height), Image.ANTIALIAS)
        start_time = time.monotonic()
        results = detect_objects(interpreter, image, 0.9)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        #print(image.size)
        
        

        annotator.clear()
        annotate_objects(annotator, results, labels)
        annotator.text([5, 0], '%.1fms' % (elapsed_ms))
        annotator.update()
        
        ymin, xmin, ymax, xmax, score = get_best_box_param(results,CAMERA_WIDTH,CAMERA_HEIGHT)
        
        if score > 0.99:
            #print(ymin, " ", xmin, " ", ymax, " ", xmax)
            #print(image_large.size)
            img = np.array(image_large)
            #print("img: ", img.shape)
            img_cut = img[ymin:ymax,xmin:xmax,:]
            print(img_cut.shape)
            img_cut = cv2.resize(img_cut, dsize=(96, 96), interpolation=cv2.INTER_CUBIC).astype('uint8')
            img_cut_pil = Image.fromarray(img_cut)
            img_cut_pil.save('scanned_people/' + str(person_number) + '/png/img_' +  str(count_images_saved) + '.png')
            np.save('scanned_people/' + str(person_number) + '/npy/img_' +  str(count_images_saved),img_cut)
            count_images_saved = count_images_saved + 1
            
        stream.seek(0)
        stream.truncate()

    finally:
      camera.stop_preview()


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def get_best_box_param(results,CAMERA_WIDTH, CAMERA_HEIGHT):
    best_boxvalue = 0
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    for obj in results:
        if obj['score'] > best_boxvalue:
            best_boxvalue = obj['score']
            ymin, xmin, ymax, xmax = obj['bounding_box']
            if xmin < 0:
                xmin = 0
            if xmax > 1:
                xmax = 1
            if ymin < 0:
                ymin = 0
            if ymax > 1:
                ymax = 1
            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)
    #print("score: ", best_boxvalue)
    return ymin, xmin, ymax, xmax, best_boxvalue


def annotate_objects(annotator, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)

    # Overlay the box, label, and score on the camera preview
    annotator.bounding_box([xmin, ymin, xmax, ymax])
    annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))




if __name__ == '__main__':
  main()

