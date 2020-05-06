"""
Run to recognize people that were scanned previously. When a person with name "X" is recognized, the program will give an
auidio output saying "Hello X".

If you do not have an Edge TPU or you want to see the performance difference, change the
variable ifEdgeTPU_1_else_0 in main() to 0.
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import io
import re
import os
import time
from tflite_runtime.interpreter import load_delegate

from annotation import Annotator

import numpy as np
import picamera

from PIL import Image
from PIL import ImageDraw
import cv2

import pyttsx3
engine = pyttsx3.init()



from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

def main():

  ifEdgeTPU_1_else_0 = 1
  
  labels = load_labels('coco_labels.txt')
  people_lables = load_labels('people_labels.txt')
  
  if ifEdgeTPU_1_else_0 == 1:
      interpreter = Interpreter(model_path = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  else:
      interpreter = Interpreter(model_path = 'models/ssd_mobilenet_v2_face_quant_postprocess.tflite')
  
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
  
  if ifEdgeTPU_1_else_0 == 1:
      interpreter_emb = Interpreter(model_path = 'models/Mobilenet1_triplet1588469968_triplet_quant_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
  else:
      interpreter_emb = Interpreter(model_path = 'models/Mobilenet1_triplet1588469968_triplet_quant.tflite')

  interpreter_emb.allocate_tensors()

  with picamera.PiCamera(
      resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
      #resolution=(320, 320), framerate=30) as camera:
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
        results = detect_objects(interpreter, image, 0.5)
        elapsed_ms = (time.monotonic() - start_time) * 1000
        #print(image.size)

        annotator.clear()
        annotate_objects(annotator, results, labels)
        annotator.text([5, 0], '%.1fms' % (elapsed_ms))
        annotator.update()
        
        ymin, xmin, ymax, xmax, score = get_best_box_param(results,CAMERA_WIDTH,CAMERA_HEIGHT)
        
        if score > 0.96:
            #print(ymin, " ", xmin, " ", ymax, " ", xmax)
            #print(image_large.size)
            img = np.array(image_large)
            #print("img: ", img.shape)
            #img = np.asarray(image_large).reshape(CAMERA_WIDTH,CAMERA_HEIGHT,3)
            #print(img.shape)
            #plt.imshow(img)
            #plt.show()
            img_cut = img[ymin:ymax,xmin:xmax,:]
            #print(img_cut.shape)
            img_cut = cv2.resize(img_cut, dsize=(96, 96), interpolation=cv2.INTER_CUBIC).astype('uint8')
            img_cut = img_cut.reshape(1,96,96,3)/255.
            #emb = FRmodel.predict(img_cut)
            emb = img_to_emb(interpreter_emb,img_cut)
            get_person_from_embedding(people_lables,emb)
            

        stream.seek(0)
        stream.truncate()

    finally:
      camera.stop_preview()

def get_person_from_embedding(people_lables,emb):
    num_emb_check = 20
    path = 'scanned_people/'
    folders = os.listdir(path)
    folders = sorted(folders)
    averages = np.zeros(len(folders))
    folder_number = 0
    start = time.time()
    for folder in folders:
        average_one_person = 0
        #print(folder)
        files = os.listdir(path + folder + '/embeddings')
        files = sorted(files)
        checked = 0
        for file in files:
            emb2 = np.load(path + folder + '/embeddings' + '/' + file)
            #print(emb.shape)
            norm = np.sum((emb-emb2)**2)
            average_one_person = average_one_person + norm
            #print(norm)
            checked = checked + 1
            if checked == num_emb_check:
                break
        average_one_person = average_one_person/num_emb_check
        averages[folder_number] = averages[folder_number] + average_one_person
        folder_number = folder_number + 1
    who_is_on_pic = 0
    lowest_norm_found = 10
    run = 0
    end = time.time()
    print("time for detection: ", end-start)
    for average in averages:
        run = run + 1
        if average < 0.6 and average < lowest_norm_found:
            lowest_norm_found = average
            who_is_on_pic = run
        print(average)
    print("person on pic: ", people_lables[who_is_on_pic])
    if who_is_on_pic > 0:
        engine.say('Hello ')
        engine.say(str(people_lables[who_is_on_pic]))
        engine.runAndWait()

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

def set_input_tensor_emb(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    scale, zero_point = input_details['quantization']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(input/scale + zero_point)



def img_to_emb(interpreter,input):
    set_input_tensor_emb(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    #emb = np.squeeze(interpreter.get_tensor(output_details['index']))
    emb = interpreter.get_tensor(output_details['index'])
    scale, zero_point = output_details['quantization']
    emb = scale * (emb - zero_point)
    return emb

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
    #annotator.text([150,300],"Felix")
    

    





if __name__ == '__main__':
  main()
