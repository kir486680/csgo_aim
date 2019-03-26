import tensorflow
import numpy as np

import numpy as np
import pyautogui
import time
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





def Shoot(mid_x, mid_y, curr_delta):
  width = 800
  height = 640
  #x, y = pyautogui.position()
  #print("Current pos" , x, y)
  x = int(mid_x*width)
  #y = int((mid_y*height))
  print(curr_delta)
  y = int((mid_y*height+height/9)+curr_delta)
  #print("Moving")
  #pyautogui.moveTo(x,y)
  #print("Middle points" , x, y)
  pyautogui.moveTo(x,y)
  time.sleep(1)

def find_boxes(image_np, image_np_expanded, sess, detection_graph, label_map, category_index):
	

	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	
	(boxes, scores, classes, num_detections) = sess.run(
	[boxes, scores, classes, num_detections],
	feed_dict={image_tensor: image_np_expanded})
	
	return boxes, scores, classes, num_detections

def shoot_decision(team, array_ch, array_c, array_th,array_t, curr_delta):
  if team == "c":
    if len(array_ch) > 0:
      Shoot(array_ch[0][0], array_ch[0][1], curr_delta)
    if len(array_ch) == 0 and len(array_c) > 0:
      Shoot(array_c[0][0], array_c[0][1], curr_delta)
  if team == "t":
    if len(array_th) > 0:
      Shoot(array_th[0][0], array_th[0][1], curr_delta )
    if len(array_th) == 0 and len(array_t) > 0:
      Shoot(array_t[0][0], array_t[0][1], curr_delta)