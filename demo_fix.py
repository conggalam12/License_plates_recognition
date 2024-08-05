import argparse
import torch
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np
import math
import torch.nn.functional as F

def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

# detect character and number in license plate
def read_plate(tensor,class_name):
    LP_type = "1"
    results = tensor
    bb_list = results.tolist()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        print(tensor)
        return "unknown"
    center_list = []
    y_mean = 0
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c,y_c,bb[-1]])

    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            id = class_name[int(l1[2])]
            license_plate += id
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            id = class_name[int(l2[2])]
            license_plate += id
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            id = class_name[int(l[2])]
            license_plate += id
    return license_plate

def convert_tensor_image(tensor):
    img = tensor.cpu().numpy()
    
    if img.shape[0] == 1:
        img = img.squeeze(0)
    
    img = img.transpose(1, 2, 0)
    

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img
def letterbox_tensor(tensor, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):

    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Input shape (1, 3, H, W)")

    _, _, h, w = tensor.shape
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(h * r)), int(round(w * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = new_shape
        r = new_shape[1] / w, new_shape[0] / h

    dw /= 2
    dh /= 2


    tensor = F.interpolate(tensor, size=new_unpad, mode='bilinear', align_corners=False)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    tensor = F.pad(tensor, (top,bottom,left,right), mode='constant', value=114/255)

    return tensor, (r, r), (dw, dh)
def run(
    weights_detect='weights/LP_detector.pt',
    weights_ocr = 'weights/LP_ocr.pt',
    source='test_image/demo3.jpg',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='cpu',
    classes=None,
    agnostic_nms=False,
    line_thickness=3
):
    # Initialize
    device = select_device(device)
    model_detect = DetectMultiBackend(weights_detect, device=device)
    model_ocr = DetectMultiBackend(weights_ocr, device=device)
    stride, names, pt = model_detect.stride, model_detect.names, model_detect.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Load image
    img_source = cv2.imread(source)
    shape_img_source = img_source.shape
    img = letterbox(img_source, imgsz, stride=stride, auto=pt)[0]
    shape_img = img.shape
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = torch.from_numpy(img.copy()).to(device)
    img = img.half() if model_detect.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]
 # HWC to CHW, BGR to RGB

    pred = model_detect(img, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process detections
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size

            for box in det:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cropped_tensor = img[:,:, y1:y2, x1:x2]
                cropped_tensor = letterbox_tensor(cropped_tensor,imgsz, stride=stride, auto=pt)[0]
                pred_ocr = model_ocr(cropped_tensor, augment=False, visualize=False)
                pred_ocr = non_max_suppression(pred_ocr, 0.7, iou_thres, classes, agnostic_nms, max_det=max_det)
                lp = read_plate(pred_ocr[0],model_ocr.names)
                # draw box
                if lp != "unknown":
                    x1 = int(x1*shape_img_source[0]/shape_img[0])
                    x2 = int(x2*shape_img_source[0]/shape_img[0])+20
                    y1 = int(y1*shape_img_source[1]/shape_img[1])-20
                    y2 = int(y2*shape_img_source[1]/shape_img[1])+10
                    cv2.rectangle(img_source, (x1,y1), (x2,y2), color = (0,0,225), thickness = 2)
                    cv2.putText(img_source, lp, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Show image
    # cv2.imshow('Result', img_source)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("result_image/demo3.jpg",img_source)

if __name__ == '__main__':
    run()