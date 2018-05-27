import torch
import numpy as np
from net.utils.func_utils import to_tensor
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rpn.rpn_utils import rpn_encode


def make_one_rpn_target(cfg, anchor_boxes, truth_boxes):
    """
    labeling windows for one image
    :param image: input image
    :param anchor_boxes: [[x0, y0, x1, y1]]: (N, 4) ndarray of float32
    :param truth_boxes:  [[x0, y0, x1, y1]]: (N, 4) ndarray of float32
    :param truth_labels: [1, 1, 1, ...], (N, ) ndarray of int64
    :return:
        anchor_labels: 1 for pos, 0 for neg
        anchor_assigns: which truth box is assigned to the anchor box
        label_weight: pos=1, neg \in (0, 1] by rareness, otherwise 0 (don't care)
        delta: bboxes' offsets
        delta_weight: same as label_weight
    """
    num_anchor_boxes = len(anchor_boxes)
    anchor_labels  = np.zeros((num_anchor_boxes,), np.int64)
    anchor_assigns = np.zeros((num_anchor_boxes,), np.int64)
    label_weight   = np.ones((num_anchor_boxes,), np.float32)  # <todo> why use 1 for init ?
    delta          = np.zeros((num_anchor_boxes, 4), np.float32)
    delta_weight   = np.zeros((num_anchor_boxes,), np.float32)

    num_truth_box = len(truth_boxes)
    if num_truth_box != 0:

        overlap = cython_box_overlap(anchor_boxes, truth_boxes)
        argmax_overlap = np.argmax(overlap, 1)
        max_overlap = overlap[np.arange(num_anchor_boxes), argmax_overlap]
        # anchor_labels 1/0 for each anchor
        bg_index = max_overlap < cfg.rpn_train_bg_thresh_high
        anchor_labels[bg_index] = 0
        label_weight[bg_index] = 1

        fg_index = max_overlap >= cfg.rpn_train_fg_thresh_low
        anchor_labels[fg_index] = 1
        label_weight[fg_index] = 1
        anchor_assigns[...] = argmax_overlap

        # for each truth, anchor_boxes with highest overlap, include multiple maxs
        # re-assign less overlapped gt to anchor_boxes
        argmax_overlap = np.argmax(overlap, 0)
        max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]
        anchor_assignto_gt, gt_assignto_anchor = np.where(overlap == max_overlap)

        fg_index = anchor_assignto_gt
        anchor_labels[fg_index] = 1
        label_weight[fg_index] = 1
        anchor_assigns[fg_index] = gt_assignto_anchor

        # regression
        fg_index = np.where(anchor_labels != 0)
        target_window = anchor_boxes[fg_index]
        target_truth_box = truth_boxes[anchor_assigns[fg_index]]
        delta[fg_index] = rpn_encode(target_window, target_truth_box)
        delta_weight[fg_index] = 1

        # weights for class balancing
        fg_index = np.where((label_weight != 0) & (anchor_labels != 0))[0]
        bg_index = np.where((label_weight != 0) & (anchor_labels == 0))[0]

        num_fg = len(fg_index)
        num_bg = len(bg_index)
        label_weight[fg_index] = 1
        label_weight[bg_index] = num_fg / num_bg

        # task balancing
        delta_weight[fg_index] = label_weight[fg_index]

    # save
    anchor_labels  = to_tensor(anchor_labels,  cfg.device)
    anchor_assigns = to_tensor(anchor_assigns, cfg.device)
    label_weight   = to_tensor(label_weight,   cfg.device)
    delta          = to_tensor(delta,          cfg.device)
    delta_weight   = to_tensor(delta_weight,   cfg.device)

    return anchor_labels, anchor_assigns, label_weight, delta, delta_weight


def make_rpn_target(cfg, anchor_boxes, truth_boxes_batch):
    """
    append -> concat -> to tensor
    :param cfg:
    :param images:
    :param anchor_boxes: list of ndarray [B*(N, 4)]
    :param truth_boxes_batch: list of ndarray [B*(N, 4)]
    :return:
        anchor_labels    (B, N) IntTensor
        anchor_label_assigns
        anchor_label_weights
        anchor_targets
        anchor_targets_weights
    """
    anchor_labels = []
    anchor_label_assigns = []
    anchor_label_weights = []
    anchor_targets = []
    anchor_targets_weights = []

    batch_size = len(truth_boxes_batch)
    for b in range(batch_size):
        truth_boxes = truth_boxes_batch[b]

        label, label_assign, label_weight, target, targets_weight = \
            make_one_rpn_target(cfg, anchor_boxes, truth_boxes)

        anchor_labels.append(label.view(1, -1))
        anchor_label_assigns.append(label_assign.view(1, -1))
        anchor_label_weights.append(label_weight.view(1, -1))
        anchor_targets.append(target.view(1, -1, 4))
        anchor_targets_weights.append(targets_weight.view(1, -1))

    anchor_labels = torch.cat(anchor_labels, 0)
    anchor_label_assigns = torch.cat(anchor_label_assigns, 0)
    anchor_label_weights = torch.cat(anchor_label_weights, 0)
    anchor_targets = torch.cat(anchor_targets, 0)
    anchor_targets_weights = torch.cat(anchor_targets_weights, 0)

    return anchor_labels, anchor_label_assigns, anchor_label_weights, anchor_targets, anchor_targets_weights
