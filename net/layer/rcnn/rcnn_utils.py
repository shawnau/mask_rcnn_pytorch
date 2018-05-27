import numpy as np
import torch
from torch.nn import functional as F


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


def rcnn_reg_loss(labels, deltas, targets, deltas_sigma=1.0):
    """
    :param labels: (\sum B_i*num_proposals_i, )
    :param deltas: (\sum B_i*num_proposals_i, num_classes*4)
    :param targets:(\sum B_i*num_proposals_i, 4)
    :param deltas_sigma: float
    :return: float
    """
    batch_size, num_classes_mul_4 = deltas.size()
    num_classes = num_classes_mul_4 // 4
    deltas = deltas.view(batch_size, num_classes, 4)

    num_pos_proposals = len(labels.nonzero())
    if num_pos_proposals > 0:
        # one hot encode. select could also seen as mask matrix
        select = torch.zeros((batch_size, num_classes)).to(deltas.device)
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0  # bg is 0, label starts from 1

        select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, 4)).contiguous().byte()
        deltas = deltas[select].view(-1, 4)

        deltas_sigma2 = deltas_sigma * deltas_sigma
        return F.smooth_l1_loss(deltas * deltas_sigma2, targets * deltas_sigma2,
                                size_average=False) / deltas_sigma2 / num_pos_proposals
    else:
        return torch.tensor(0.0, requires_grad=True).to(deltas.device)


def rcnn_cls_loss(logits, labels):
    """
    :param logits: (\sum B_i*num_proposals_i, num_classes)
    :param labels: (\sum B_i*num_proposals_i, )
    :return: float
    """
    return F.cross_entropy(logits, labels, size_average=True)