import numpy as np
import torch
from net.utils.func_utils import np_softmax
from net.utils.box_utils import clip_boxes, filter_boxes
from net.lib.nms.cython_nms import cython_nms


def rcnn_encode(bboxes, targets):
    """
    :param bboxes: bboxes
    :param targets: target ground truth boxes
    :return: deltas
    """

    bw = bboxes[:, 2] - bboxes[:, 0] + 1.0
    bh = bboxes[:, 3] - bboxes[:, 1] + 1.0
    bx = bboxes[:, 0] + 0.5 * bw
    by = bboxes[:, 1] + 0.5 * bh

    tw = targets[:, 2] - targets[:, 0] + 1.0
    th = targets[:, 3] - targets[:, 1] + 1.0
    tx = targets[:, 0] + 0.5 * tw
    ty = targets[:, 1] + 0.5 * th

    dx = (tx - bx) / bw
    dy = (ty - by) / bh
    dw = np.log(tw / bw)
    dh = np.log(th / bh)

    deltas = np.vstack((dx, dy, dw, dh)).transpose()
    return deltas


def rcnn_decode(bboxes, deltas):
    """
    :param bboxes: bounding boxes
    :param deltas: bbox regression deltas
    :return: refined bboxes
    """
    num = len(bboxes)
    predictions = np.zeros((num, 4), dtype=np.float32)
    # if num == 0: return predictions  #not possible?

    bw = bboxes[:, 2] - bboxes[:, 0] + 1.0
    bh = bboxes[:, 3] - bboxes[:, 1] + 1.0
    bx = bboxes[:, 0] + 0.5 * bw
    by = bboxes[:, 1] + 0.5 * bh
    bw = bw[:, np.newaxis]
    bh = bh[:, np.newaxis]
    bx = bx[:, np.newaxis]
    by = by[:, np.newaxis]

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    x = dx * bw + bx
    y = dy * bh + by
    dw = np.clip(dw, -10, 10)
    dh = np.clip(dh, -10, 10)
    w = np.exp(dw) * bw
    h = np.exp(dh) * bh

    predictions[:, 0::4] = x - 0.5 * w  # x0,y0,x1,y1
    predictions[:, 1::4] = y - 0.5 * h
    predictions[:, 2::4] = x + 0.5 * w  # todo: should there be a -1 ?
    predictions[:, 3::4] = y + 0.5 * h  # todo: should there be a -1 ?

    return predictions

def rcnn_nms(cfg, mode, images, rpn_proposals, logits, deltas):
    """
    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param images: a batch of input images
    :param rpn_proposals: rpn proposals (N, ) [i, x0, y0, x1, y1, score, label]
    :param logits:
    :param deltas:
    :return:
        [i, x0, y0, x1, y1, score, label]
    """
    if mode in ['train']:
        nms_prob_threshold = cfg.rcnn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_train_nms_overlap_threshold
        nms_min_size = cfg.rcnn_train_nms_min_size

    elif mode in ['valid', 'test', 'eval']:
        nms_prob_threshold = cfg.rcnn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_test_nms_overlap_threshold
        nms_min_size = cfg.rcnn_test_nms_min_size

        if mode in ['eval']:
            nms_prob_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)

    num_classes = cfg.num_classes
    rpn_proposals = rpn_proposals.detach().numpy()  # todo: do we need to detach?
    logits = logits.detach().numpy()
    deltas = deltas.detach().numpy().reshape(-1, num_classes, 4)
    batch_size, _, height, width = images.size()

    # non-max suppression
    rcnn_proposals = []
    for img_idx in range(batch_size):  # image index in a batch
        pic_proposals = [np.empty((0, 7), np.float32)]
        select = np.where(rpn_proposals[:, 0] == img_idx)[0]
        if len(select) == 0:
            return torch.from_numpy(np.vstack(np.empty((0, 7), np.float32))).to(cfg.device)
        raw_box = rpn_proposals[select, 1:5]
        prob_distrib  = np_softmax(logits[select])  # <todo>why not use np_sigmoid?
        delta_distrib = deltas[select]

        # skip background
        for cls_idx in range(1, num_classes):
            index = np.where(prob_distrib[:, cls_idx] > nms_prob_threshold)[0]
            if len(index) > 0:
                valid_box = raw_box[index]
                prob = prob_distrib[index, cls_idx].reshape(-1, 1)
                delta = delta_distrib[index, cls_idx]
                # bbox regression, clip & filter
                box = rcnn_decode(valid_box, delta)
                box = clip_boxes(box, width, height)
                keep = filter_boxes(box, min_size=nms_min_size)

                if len(keep) > 0:
                    box = box[keep]
                    prob = prob[keep]
                    keep = cython_nms(np.hstack((box, prob)), nms_overlap_threshold)

                    detection = np.zeros((len(keep), 7), np.float32)
                    detection[:, 0] = img_idx
                    detection[:, 1:5] = np.around(box[keep], 0)
                    detection[:, 5] = prob[keep, 0]  # p[:, 0]
                    detection[:, 6] = cls_idx
                    pic_proposals.append(detection)

        pic_proposals = np.vstack(pic_proposals)
        rcnn_proposals.append(pic_proposals)

    rcnn_proposals = torch.from_numpy(np.vstack(rcnn_proposals)).to(cfg.device)
    return rcnn_proposals