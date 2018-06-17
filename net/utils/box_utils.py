import numpy as np
import torch
import cv2
from skimage.transform import resize


# <todo> mask crop should match align kernel (same wait to handle non-integer pixel location (e.g. 23.5, 32.1))
def resize_instance(instance, box, mask_size, threshold=0.5):
    """
    return the ground truth mask for mask head
    :param instance: one mask of (H, W) of input image, e.g.
        [[0, 1, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
        for a 3x3 image
    :param box: bbox on input image. e.g.
        [x0, y0, x1, y1]
    :param mask_size: mask_size of the output of maskhead. e.g. 28*28
    :param threshold: used to define pos/neg pixels of the mask
    :return: cropped & resized mask into mask_size of the mask head output
    """
    H, W = instance.shape
    x0, y0, x1, y1 = np.rint(box).astype(np.int32)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    #<todo> filter this
    if 1:
        if x0 == x1:
            x0 = x0-1
            x1 = x1+1
            x0 = max(0, x0)
            x1 = min(W, x1)
        if y0 == y1:
            y0 = y0-1
            y1 = y1+1
            y0 = max(0, y0)
            y1 = min(H, y1)

    #print(x0,y0,x1,y1)
    crop = instance[y0:y1+1,x0:x1+1]
    # crop = cv2.resize(crop, (mask_size, mask_size))
    crop = resize(crop, (mask_size, mask_size))
    crop = (crop > threshold).astype(np.float32)
    return crop

def torch_clip_proposals(proposals, index, width, height):
    proposals = torch.stack((
        proposals[index, 0],
        proposals[index, 1].clamp(0, width - 1),
        proposals[index, 2].clamp(0, height - 1),
        proposals[index, 3].clamp(0, width - 1),
        proposals[index, 4].clamp(0, height - 1),
        proposals[index, 5],
        proposals[index, 6],
    ), 1)
    return proposals


# python
def clip_boxes(boxes, width, height):
    """
    Clip process to image boundaries.
    Used in rpn_nms and rcnn_nms
    :param boxes: proposals
    :param width: input's_train width
    :param height: input's_train height
    :return: cropped proposals to fit input's_train border
    """
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def is_small_box_at_boundary(box, W, H, min_size):
    x0, y0, x1, y1 = box
    w = (x1 - x0) + 1
    h = (y1 - y0) + 1
    aspect = max(w, h) / min(w, h)
    area = w * h
    return ((x0 == 0 or x1 == W - 1) and (w < min_size)) or \
           ((y0 == 0 or y1 == H - 1) and (h < min_size))


def is_small_box(box, min_size):
    x0, y0, x1, y1 = box
    w = (x1 - x0) + 1
    h = (y1 - y0) + 1
    aspect = max(w, h) / min(w, h)
    area = w * h
    return (w < min_size or h < min_size)


def is_big_box(box, max_size):
    x0, y0, x1, y1 = box
    w = (x1 - x0) + 1
    h = (y1 - y0) + 1
    aspect = max(w, h) / min(w, h)
    area = w * h
    return (w > max_size or h > max_size)
