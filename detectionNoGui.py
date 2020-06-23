# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import mss
import pyautogui
import time


# Initialize the parameters
confThreshold = 0.4  #Confidence threshold
nmsThreshold = 0.6   #Non-maximum suppression threshold
inpWidth = 416     #Width of network's input image
inpHeight = 416      #Height of network's input image

pyautogui.FAILSAFE = False

width = 1920
height = 1080

monitor = {"top": 80, "left": 0, "width": width, "height": height}
#0 ,1 for Terrorist, Terrorist Head and 2,3 for Counter Terrorist and CT head
friendlyTeam = [2,3]

sct = mss.mss()
# Load names of classes
classesFile = "obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny_last.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



def Shoot(mid_x, mid_y):
  print(mid_x, mid_y)
  x = int(mid_x*width)
  #y = int(mid_y*height)
  y = int(mid_y*height+height/9)
  pyautogui.moveTo(mid_x,mid_y+70)
  #pyautogui.click()
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                if classId not in friendlyTeam:
                    Shoot(center_x, center_y)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        if classIds[i] not in friendlyTeam:
            Shoot(center_x, center_y)
# Process inputs
winName = 'CSGO ObjectDetection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)


while True:
    
    # get frame from the video
    
    image_np = np.array(sct.grab(monitor))
      # To get real color we do this:
    frame = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)







