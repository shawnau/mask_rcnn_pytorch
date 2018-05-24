import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net.lib.roi_align.module import RoIAlign as Crop


# https://qiita.com/yu4u/items/5cbe9db166a5d72f9eb8
class CropRoi(nn.Module):
    def __init__(self, cfg, crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.num_scales = len(cfg.rpn_scales)
        self.crop_size  = crop_size
        self.sizes      = cfg.rpn_base_sizes
        self.scales     = cfg.rpn_scales

        self.crops = nn.ModuleList()
        for l in range(self.num_scales):
            self.crops.append(
                Crop(self.crop_size, self.crop_size, 1/self.scales[l])
            )

    def forward(self, fs, proposals):
        """
        :param fs: [p2, p3, p4, p5]
        :param proposals: bbox filtered by nms
            [i, x0, y0, x1, y1, score, label]
        :return:
            cropped feature maps for each box
        """
        num_proposals = len(proposals)

        # this is complicated. we need to decide for a given roi,
        # which of the p2,p3,p4,p5 layers to pool from
        boxes = proposals.detach().data[:, 1:5]
        sizes = boxes[:, 2:]-boxes[:, :2]  # box sizes (w, h)
        sizes = torch.sqrt(sizes[:, 0]*sizes[:, 1])  # sqrt(wh)
        # compare box with 4 different base sizes
        distances = torch.abs(sizes.view(num_proposals, 1).expand(num_proposals, 4) -
                              torch.from_numpy(np.array(self.sizes, np.float32)).to(self.cfg.device))
        #     sizes: 8, 16, 32, 64
        # distances: 1,  3,  3,  7 (min_index=0, pool from fs[0]=p2)
        min_distances, min_index = distances.min(1)

        rois = proposals.detach().data[:, 0:5]
        rois = Variable(rois)
        # pool from each layer in fs
        crops   = []
        indices = []
        for l in range(self.num_scales):
            index = (min_index == l).nonzero()
            if len(index) > 0:
                crop = self.crops[l](fs[l], rois[index].view(-1, 5))
                # crop = self.crops[l](fs[l], proposals[index].view(-1, 8))
                crops.append(crop)
                indices.append(index)
        # re-arrange crops by box order
        crops   = torch.cat(crops, 0)
        indices = torch.cat(indices, 0).view(-1)
        crops   = crops[torch.sort(indices)[1]]
        # crops = torch.index_select(crops,0,index)

        return crops