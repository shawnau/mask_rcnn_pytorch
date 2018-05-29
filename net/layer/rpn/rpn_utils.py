import itertools
import numpy as np
from net.utils.func_utils import weighted_focal_loss_for_cross_entropy, weighted_smooth_l1


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
    :param: fs: [4*(B, C, H, W)] a batch of features
    create region proposals from all 4 feature maps from FPN
    a total of (128*128*3 + 64*64*3 + 32*32*3 + 16*16*3) boxes
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


def rpn_cls_loss(logits, labels, label_weights):
    """
    :param logits: (B, N, 2),    unnormalized foreground/background score
    :param labels: (B, N)        {0, 1} for bg/fg
    :param label_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :return: float
    """
    batch_size, num_anchors, num_classes = logits.size()
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # classification ---
    logits = logits.view(batch_num_anchors, num_classes)
    labels = labels.view(batch_num_anchors, 1)
    label_weights = label_weights.view(batch_num_anchors, 1)

    return weighted_focal_loss_for_cross_entropy(logits, labels, label_weights)


def rpn_reg_loss(labels, deltas, target_deltas, target_weights, delta_sigma=3.0):
    """
    :param labels: (B, N) {0, 1} for bg/fg. used to catch positive samples
    :param deltas: (B, N, 2, 4) bbox regression
    :param target_deltas: (B, N, 4) target deltas from make_rpn_target
    :param target_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :param delta_sigma: float
    :return: float
    """
    batch_size, num_anchors, num_classes, num_deltas = deltas.size()
    assert num_deltas == 4
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # one-hot encode
    labels = labels.view(batch_num_anchors, 1)
    deltas = deltas.view(batch_num_anchors, num_classes, 4)
    target_deltas = target_deltas.view(batch_num_anchors, 4)
    target_weights = target_weights.view(batch_num_anchors, 1)
    # calc positive samples only
    if len((labels != 0).nonzero()) > 0:
        index = (labels != 0).nonzero()[:, 0]
        deltas = deltas[index]
        target_deltas = target_deltas[index]
        target_weights = target_weights[index].expand((-1, 4)).contiguous()

        select = labels[index].view(-1, 1).expand((-1, 4)).contiguous().view(-1, 1, 4)
        deltas = deltas.gather(1, select).view(-1, 4)
    else:  # all empty!
        index = (labels != 0).nonzero()
        deltas = deltas[index]
        target_deltas = target_deltas[index]
        target_weights = target_weights[index]

    return weighted_smooth_l1(deltas, target_deltas, target_weights, delta_sigma)