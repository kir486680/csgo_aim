import argparse
import mss
from utils.datasets import *
from utils.utils import *
import cv2 as cv
import constants as consts
import logging
from datetime import datetime
monitor = {"top": 80, "left": 0, "width": consts.width, "height": consts.height}
sct = mss.mss()
print(type(consts.friendlyTeam))
logging.basicConfig(filename='info.log', level=logging.INFO)
def detect(save_img=False):
    half = False
    # Initialize
    try:
        device = torch_utils.select_device(consts.device)
        weights = consts.modelWeights
        conf = consts.confThreshold
        model = torch.load(weights, map_location=device)['model']
        # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
        # model.fuse()
        model.to(device).eval()
        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()
        # Get names and colors
        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    except:
        logging.info(str(datetime.now()) + " Failed to initialize the model")
    else:
        logging.info(str(datetime.now()) + " Model is initialized")

    while True:
            image_np = np.array(sct.grab(monitor))
            # To get real color we do this:
            frame = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)

            img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
            #cv2.imshow('ImageWindow', img)
            #cv2.waitKey()
            #img = cv2.imread("sample_img/bopencv_frame_c100.jpg") 
            im0 = img
            print(img.shape)
            img = letterbox(img, new_shape=416)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            print(img.shape)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)


            pred = model(img, False)[0]

            # to float
            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5,
                                   fast=True, classes=None, agnostic=False)
            for i, det in enumerate(pred): 
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *x, conf, cls in det:
                        if cls.int() not in consts.friendlyTeam:
                            pyautogui.moveTo((int(x[0])+int(x[2]))/2, (int(x[1])+int(x[3]))/2)
                        #plot_one_box(x, im0, label=label, color=colors[int(cls)], line_thickness=3)






if __name__ == '__main__':
    with torch.no_grad():
        detect()
