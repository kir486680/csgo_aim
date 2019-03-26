import time
import cv2
import mss
import numpy as np
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import pyautogui
import win32api
import keyboard
#from GUI import 
from helper import find_boxes, Shoot, shoot_decision

from tkinter import *

master = Tk()
canvas = Canvas(master, width=30, height=30, bd=0, highlightthickness=0)
canvas.pack()
v = StringVar()
label = Label(master, textvariable=v)
label.pack()
w = Scale(master, from_=-50, to=50)
w.pack(side=TOP, anchor=W, fill=X, expand=YES)



team = "t"

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
width = 800
height = 640

monitor = {"top": 80, "left": 0, "width": width, "height": height}




# ## Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# # Model preparation 
PATH_TO_FROZEN_GRAPH = 'CSGO_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'CSGO_labelmap.pbtxt'
NUM_CLASSES = 4


# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# # Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      try:
        master.update_idletasks()
        master.update() 
        curr_delta = w.get()
      except:
        print("Done")
        sys.exit(0)
      

      image_np = np.array(sct.grab(monitor))
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      image_np_expanded = np.expand_dims(image_np, axis=0)

      

      (boxes, scores, classes, num_detections) = find_boxes(image_np, image_np_expanded, sess, detection_graph , label_map, category_index) 
      array_ch = []
      array_c = []
      array_th = []
      array_t = []
      for i,b in enumerate(boxes[0]):
        if classes[0][i] == 2: # ch
          if scores[0][i] >= 0.8:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            array_ch.append([mid_x, mid_y])
            cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (0,0,255), -1)
        if classes[0][i] == 1: # c 
          if scores[0][i] >= 0.8:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
            array_c.append([mid_x, mid_y])
            cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (50,150,255), -1)
        if classes[0][i] == 4: # th
          if scores[0][i] >= 0.8:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            array_th.append([mid_x, mid_y])
            cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (0,0,255), -1)
        if classes[0][i] == 3: # t
          if scores[0][i] >= 0.8:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
            array_t.append([mid_x, mid_y])
            cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (50,150,255), -1)
      if keyboard.is_pressed('alt'):  # if key 'q' is pressed 
            #print('You Pressed A Key!')
            if team=="t":
              team = "c"
              pass
            elif team=="c":
              team ="t"
            time.sleep(0.1)
      v.set(team)
      
      shoot_decision(team,  array_ch, array_c, array_th,array_t, curr_delta)
      

  
      # Show image with detection
      #cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
      # Bellow we calculate our FPS
      fps+=1
      TIME = time.time() - start_time
      if (TIME) >= display_time :
        print("FPS: ", fps / (TIME))
        fps = 0
        start_time = time.time()
      # Press "q" to quit
      #if cv2.waitKey(25) & 0xFF == ord("q"):
        #cv2.destroyAllWindows()
        #break