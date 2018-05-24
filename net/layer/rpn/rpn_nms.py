import itertools
import torch
import numpy as np
from net.utils.box_utils import clip_boxes, filter_boxes
from net.utils.func_utils import np_softmax
from net.lib.nms.cython_nms import cython_nms

# ------------------------------ bbox regression -------------------------------------
# "UnitBox: An Advanced Object Detection Network" - Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, Thomas Huang
#  https://arxiv.org/abs/1608.01471
def rpn_encode(window, truth_box):
    cx = (window[:,0] + window[:,2])/2
    cy = (window[:,1] + window[:,3])/2
    w  = (window[:,2] - window[:,0]+1)
    h  = (window[:,3] - window[:,1]+1)

    target = (truth_box - np.column_stack([cx,cy,cx,cy]))/np.column_stack([w,h,w,h])
    target = target*np.array([-1,-1,1,1],np.float32)
    return target


def rpn_decode(window, delta):
    cx = (window[:,0] + window[:,2])/2
    cy = (window[:,1] + window[:,3])/2
    w  = (window[:,2] - window[:,0]+1)
    h  = (window[:,3] - window[:,1]+1)

    delta = delta*np.array([-1,-1,1,1],np.float32)
    box   = delta*np.column_stack([w,h,w,h]) + np.column_stack([cx,cy,cx,cy])
    return box


def rpn_make_anchor_boxes(fs, cfg):
    """
    :param: fs: (num_scales, B, C, H, W) a batch of features
    create region proposals from all 4 feature maps from FPN
    total of (128*128*3 + 64*64*3 + 32*32*3 + 16*16*3) boxes
    """

    def make_bases(base_size, base_apsect_ratios):
        """
        make base anchor boxes for each base size & ratio
        :param:
            base_size:
                anchor base size
                e.g. 16
            base_apsect_ratios:
                height/width ratio
                e.g. [(1, 1), (1, 2), (2, 1)]
        :return:
            list of bases, each base has 4 coordinates, a total of
            len(base_apsect_ratios) bases. e.g.
            [[ -8.,  -8.,   8.,   8.],
             [ -8., -16.,   8.,  16.],
             [-16.,  -8.,  16.,   8.]]
        """
        bases = []
        for ratio in base_apsect_ratios:
            w = ratio[0] * base_size
            h = ratio[1] * base_size
            rw = round(w / 2)
            rh = round(h / 2)
            base = (-rw, -rh, rw, rh)
            bases.append(base)

        bases = np.array(bases, np.float32)
        return bases

    def make_anchor_boxes(f, scale, bases):
        """
        make anchor boxes on every pixel of the feature map
        :param:
            f_submit: feature of size (B, C, H, W)
            scale:
                zoom scale from feature map to image,
                used to define stride on image. e.g. 4
            bases:
                base anchor boxes. e.g.
                [[ -8.,  -8.,   8.,   8.],
                 [ -8., -16.,   8.,  16.],
                 [-16.,  -8.,  16.,   8.]]
        :return:
            list of anchor boxes on input image
            shape: H * W * len(base_apsect_ratios)
        """
        anchor_boxes = []
        _, _, H, W = f.size()
        for y, x in itertools.product(range(H), range(W)):
            cx = x * scale
            cy = y * scale
            for box in bases:
                x0, y0, x1, y1 = box
                x0 += cx
                y0 += cy
                x1 += cx
                y1 += cy
                anchor_boxes.append([x0, y0, x1, y1])

        anchor_boxes = np.array(anchor_boxes, np.float32)
        return anchor_boxes

    rpn_anchor_boxes = []
    num_scales = len(cfg.rpn_scales)
    for l in range(num_scales):
        bases = make_bases(cfg.rpn_base_sizes[l], cfg.rpn_base_apsect_ratios[l])
        boxes = make_anchor_boxes(fs[l], cfg.rpn_scales[l], bases)
        rpn_anchor_boxes.append(boxes)

    rpn_anchor_boxes = np.vstack(rpn_anchor_boxes)
    return rpn_anchor_boxes


def rpn_nms(cfg, mode, images, anchor_boxes, logits_flat, deltas_flat):
    """
    This function:
    1. Do non-maximum suppression on given window and logistic score
    2. filter small rpn_proposals, crop border
    3. bbox regression

    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param images: a batch of input images
    :param anchor_boxes: all anchor boxes in a batch, list of coords, e.g.
               [[x0, y0, x1, y1], ...], a total of 16*16*3 + 32*32*3 + 64*64*3 + 128*128*3
    :param logits_flat: (B, N, 2) NOT nomalized
               [[0.7, 0.5], ...]
    :param deltas_flat: (B, N, 2, 4)
               [[[t1, t2, t3, t4], [t1, t2, t3, t4]], ...]
    :return: all proposals in a batch. e.g.
        [i, x0, y0, x1, y1, score, label]
        proposals[0]:   image idx in the batch
        proposals[1:5]: bbox
        proposals[5]:   probability of foreground (background skipped)
        proposals[6]:   class label, 1 fore foreground, 0 for background, here we only return 1
    """
    if mode in ['train']:
        nms_prob_threshold = cfg.rpn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_train_nms_overlap_threshold
        nms_min_size = cfg.rpn_train_nms_min_size

    elif mode in ['valid', 'test', 'eval']:
        nms_prob_threshold = cfg.rpn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_test_nms_overlap_threshold
        nms_min_size = cfg.rpn_test_nms_min_size

        if mode in ['eval']:
            nms_prob_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?' % mode)

    num_classes = 2
    logits = logits_flat.detach().numpy()
    deltas = deltas_flat.detach().numpy()
    batch_size, _, height, width = images.size()

    # non-max suppression
    rpn_proposals = []
    for img_idx in range(batch_size):
        pic_proposals = [np.empty((0, 7), np.float32)]
        prob_distrib = np_softmax(logits[img_idx])  # (N, 2)
        delta_distrib = deltas[img_idx]  # (N, 2, 4)

        # skip background
        for cls_idx in range(1, num_classes):  # 0 for background, 1 for foreground
            index = np.where(prob_distrib[:, cls_idx] > nms_prob_threshold)[0]
            if len(index) > 0:
                raw_box = anchor_boxes[index]
                prob  = prob_distrib[index, cls_idx].reshape(-1, 1)
                delta = delta_distrib[index, cls_idx]
                # bbox regression, do some clip/filter
                box = rpn_decode(raw_box, delta)
                box = clip_boxes(box, width, height)  # take care of borders
                keep = filter_boxes(box, min_size=nms_min_size)  # get rid of small boxes

                if len(keep) > 0:
                    box = box[keep]
                    prob = prob[keep]
                    keep = cython_nms(np.hstack((box, prob)), nms_overlap_threshold)

                    proposal = np.zeros((len(keep), 7), np.float32)
                    proposal[:, 0] = img_idx
                    proposal[:, 1:5] = np.around(box[keep], 0)
                    proposal[:, 5] = prob[keep, 0]
                    proposal[:, 6] = cls_idx
                    pic_proposals.append(proposal)

        pic_proposals = np.vstack(pic_proposals)
        rpn_proposals.append(pic_proposals)

    rpn_proposals = torch.from_numpy(np.vstack(rpn_proposals)).to(cfg.device)
    return rpn_proposals