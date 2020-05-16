"""
Converts the detected faces in the folder "scanned_people" to embeddings, which are used to identify people.

Important: You have to execute this code for each subfolder in "scanned_people". To do so change the variable
"scan_person" in main() for each folder number.

If you do not have an Edge TPU or you want to see the performance difference, change the
variable ifEdgeTPU_1_else_0 in main() to 0.
"""

import os
import shutil
from PIL import Image
import numpy as np
import time
import os

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter


def main():
    
    ifEdgeTPU_1_else_0 = 1
    
    scan_person = 1 # Change the number of the folder, where you want to create the embeddings

    #get interpreter for face embedding model
    if ifEdgeTPU_1_else_0 == 1:
      interpreter = Interpreter(model_path = 'models/Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite',
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
      interpreter = Interpreter(model_path = 'models/Mobilenet1_triplet1589223569_triplet_quant.tflite')

    
    interpreter.allocate_tensors()

    
    path_person = 'scanned_people/' + str(scan_person)
    
    if os.path.isdir(path_person + '/embeddings') == False:
        os.mkdir(path_person + '/embeddings')
    else:
        shutil.rmtree(path_person + '/embeddings')
        os.mkdir(path_person + '/embeddings')
    
    files = os.listdir(path_person + '/npy')
    files = sorted(files)

    for file in files:
        print(file)
        img = np.load(path_person + '/npy/' + file).reshape(1,96,96,3)/255.
        #emb = FRmodel.predict(img)
        emb = img_to_emb(interpreter,img)
        np.save(path_person + '/embeddings/' + file,emb)

def set_input_tensor(interpreter, input):
    #Sets the input tensor.
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    scale, zero_point = input_details['quantization']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(input/scale + zero_point)



def img_to_emb(interpreter,input):
    #returns embedding vector, using the face embedding model
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    #emb = np.squeeze(interpreter.get_tensor(output_details['index']))
    emb = interpreter.get_tensor(output_details['index'])
    scale, zero_point = output_details['quantization']
    emb = scale * (emb - zero_point)
    return emb



        
        

if __name__ == '__main__':
  main()
