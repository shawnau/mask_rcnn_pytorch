import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.nn.modules.module import Module
from function import CropAndResizeFunction


class CropAndResize(Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)


# See more details on
# https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
class RoIAlign(Module):
    def __init__(self, crop_height, crop_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width  = crop_width
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):

        #need to normalised (x0,y0,x1,y1) to [0,1]
        height, width = features.size()[2:4]
        ids, x0, y0, x1, y1= torch.split(rois, 1, dim=1)
        ids = ids.int()
        x0 = x0*self.spatial_scale
        y0 = y0*self.spatial_scale
        x1 = x1*self.spatial_scale
        y1 = y1*self.spatial_scale

        if 1:
             x0 = x0 / (width -1)
             y0 = y0 / (height-1)
             x1 = x1 / (width -1)
             y1 = y1 / (height-1)
             boxes = torch.cat((y0, x0, y1, x1), 1)

        boxes = boxes.detach().contiguous()
        ids   = ids.detach()
        return CropAndResizeFunction(self.crop_height, self.crop_width, extrapolation_value=0)(features, boxes, ids)
