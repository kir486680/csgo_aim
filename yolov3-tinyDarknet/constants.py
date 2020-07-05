import configparser
import json
import os
import ast
path = "settings.ini"

config = configparser.ConfigParser()
config.read(path)

# Читаем некоторые значения из конфиг. файла.
confThreshold = config.getfloat("Settings", "confThreshold")
nmsThreshold = config.getfloat("Settings", "nmsThreshold")
inpWidth = config.getint("Settings", "inpWidth")
inpHeight = config.getint("Settings", "inpHeight")
modelConfiguration = config.get("ModelParams", "modelConfiguration")
modelWeights = config.get("ModelParams", "modelWeights")
classesFile = config.get("ModelParams", "classesFile")
width = config.getint("CaptureParams", "width")
height = config.getint("CaptureParams", "height")
friendlyTeam = ast.literal_eval(config.get("InGameParams", "friendlyTeam"))