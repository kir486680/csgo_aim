from tkinter import *
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import mss
import pyautogui
import time
import keyboard
import constants as consts
friendlyTeam = []
pyautogui.FAILSAFE = False
monitor = {"top": 80, "left": 0, "width": consts.width, "height": consts.height}
sct = mss.mss()
class Cheat:

    def __init__(self):
        yolo = Yolo()
        
        
        self.label  = Label(master, text="AIm Assist")
        self.label.pack()
        self.var1 = IntVar()
        self.checkBox1 = Checkbutton(master, text="T", variable=self.var1)
        self.checkBox1.pack()
        self.var2 = IntVar()
        self.checkBox2 = Checkbutton(master, text="TH", variable=self.var2)
        self.checkBox2.pack()
        self.var3 = IntVar()
        self.checkBox3 = Checkbutton(master, text="C", variable=self.var3)
        self.checkBox3.pack()
        self.var4 = IntVar()
        self.checkBox4 = Checkbutton(master, text="CH", variable=self.var4)
        self.checkBox4.pack()
        self.update_button = Button(master, text="Update", command=self.update)
        self.update_button.pack()
        self.startDetection_button = Button(master, text="Start Detection", command=yolo.start)
        self.startDetection_button.pack()

    def update(self):
        friendlyTeam.clear()
        if self.var1.get() ==1:
            friendlyTeam.append("0")
        elif "0" in friendlyTeam:
            friendlyTeam.remove("0")
        if self.var2.get() == 1:
            friendlyTeam.append("1")
        elif "1" in friendlyTeam:
            friendlyTeam.remove("1")
        if self.var3.get() == 1:
            friendlyTeam.append("2")
        elif "2" in friendlyTeam:
            friendlyTeam.remove("2")
        if self.var4.get() == 1:
            friendlyTeam.append("3")
        elif "3" in friendlyTeam:
            friendlyTeam.remove("3")
        print(friendlyTeam)
class Yolo:

    def __init__(self):
        self.classes = None
        with open(consts.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # Give the configuration and weight files for the model and load the network using them.
        self.net = cv.dnn.readNetFromDarknet(consts.modelConfiguration, consts.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # Get the names of the output layers
    def getOutputsNames(self,net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs):
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
                if confidence > consts.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, consts.confThreshold, consts.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            #Comment this out if you want to see boxes 
            if classIds[i] not in friendlyTeam:
                self.Shoot(center_x, center_y)
    def Shoot(self,mid_x, mid_y):
        pyautogui.moveTo(mid_x,mid_y+50)
    def start(self):
        while True:
            master.update_idletasks()
            master.update() 
            
            # get frame from the video
            
            image_np = np.array(sct.grab(monitor))
            # To get real color we do this:
            frame = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1/255, (consts.inpWidth, consts.inpHeight), [0,0,0], 1, crop=False)

            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames(self.net))

            # Remove the bounding boxes with low confidence
            self.postprocess(frame, outs)

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
master = Tk()
my_gui = Cheat()
master.mainloop()


