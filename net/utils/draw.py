import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_show(name, image, resize=1):
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_screen_rect(image, point1, point2, color, alpha=0.5):
    x1, y1 = point1
    x2, y2 = point2
    image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)


def draw_boxes(image, boxes, color=(0, 0, 255)):
    for box in boxes:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)


def instances_to_color_overlay(instances, image=None, color=None):

    height,width = instances.shape[1:]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = len(instances)
    if num_masks==0:
        return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_masks))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_masks) ]

    for i in range(num_masks):
        mask = instances[i]
        overlay[mask != 0] = color[i]

    return overlay

def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def instances_to_contour_overlay(instances, image=None, color=[255,255,255]):

    height,width = instances.shape[1:]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = len(instances)
    if num_masks == 0:
        return overlay

    for i in range(num_masks):
        mask = instances[i]
        contour = mask_to_inner_contour(mask)
        overlay[contour != 0] = color

    return overlay