import multiprocessing
from multiprocessing import Pipe
import time
import cv2
import mss
import numpy as np
import datetime
import tensorflow as tf

title = "FPS benchmark"
start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}



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


global sess 
sess =  tf.Session(graph=detection_graph)

def GRABMSS_screen(p_input):
    while True:
        #Grab screen image
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_expand = np.expand_dims(img, axis=0)
        # Put image from pipe
        p_input.send(img, img_expand)
    
def SHOWMSS_screen(p_output_1 , p_output_2):
    
    global fps, start_time
    while True:

        image_np , image_np_expanded = p_output.recv()
        (boxes, scores, classes, num_detections) = find_boxes(image_np, image_np_expanded, sess, detection_graph , label_map, category_index)
        
        
        
        # Display the picture
        cv2.imshow(title, img)
        
        # Calculate FPS
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
            
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        

if __name__=="__main__":
    # Pipes
    p_output_1, p_output_2, p_input = 1, 1, 1

    
    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(p_input, ))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(p_output_1, p_output_2 ))
    p1.start()
    p2.start()


