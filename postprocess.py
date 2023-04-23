from multimodality.utils.utils import rescale_boxes, non_max_suppression
import torch
import numpy as np

def postprocess(img, outputs, processor, classes, conf_thres=0.5, nms_thres=0.5):
    shape = img.shape[:2]
    trans = processor.batch_decode(np.argmax(outputs[1], axis=-1))[0].split(" ")
    print(trans)
    labels = [float(classes.index(i)) for i in trans if i in classes]
    detections = torch.tensor(outputs[0])
    detections = non_max_suppression(detections, conf_thres, nms_thres, classes=labels)
    detections = [rescale_boxes(i, 416, shape) for i in detections]
    return detections
    