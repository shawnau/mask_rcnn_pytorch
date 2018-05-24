import torch
import numpy as np
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rpn.rpn_utils import rpn_encode


def make_one_rpn_target(cfg, anchor_boxes, truth_boxes):
    """
    labeling windows for one image
    :param image: input image
    :param anchor_boxes: list of bboxes e.g. [x0, y0, x1, y1]
    :param truth_boxes: list of boxes, e.g. [x0, y0, x1, y1]
    :param truth_labels: 1 for sure
    :return:
        label: 1 for pos, 0 for neg
        label_assign: which truth box is assigned to the window
        label_weight: pos=1, neg \in (0, 1] by rareness, otherwise 0 (don't care)
        target: bboxes' offsets
        target_weight: same as label_weight
    """
    num_anchor_boxes = len(anchor_boxes)
    label = np.zeros((num_anchor_boxes,), np.float32)
    label_assign = np.zeros((num_anchor_boxes,), np.int32)
    label_weight = np.ones((num_anchor_boxes,), np.float32)  # <todo> why use 1 for init ?
    target = np.zeros((num_anchor_boxes, 4), np.float32)
    target_weight = np.zeros((num_anchor_boxes,), np.float32)

    num_truth_box = len(truth_boxes)
    if num_truth_box != 0:

        overlap = cython_box_overlap(anchor_boxes, truth_boxes)
        argmax_overlap = np.argmax(overlap, 1)
        max_overlap = overlap[np.arange(num_anchor_boxes), argmax_overlap]
        # label 1/0 for each anchor
        bg_index = max_overlap < cfg.rpn_train_bg_thresh_high
        label[bg_index] = 0
        label_weight[bg_index] = 1

        fg_index = max_overlap >= cfg.rpn_train_fg_thresh_low
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap

        # for each truth, anchor_boxes with highest overlap, include multiple maxs
        # re-assign less overlapped gt to anchor_boxes
        argmax_overlap = np.argmax(overlap, 0)
        max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]
        anchor_assignto_gt, gt_assignto_anchor = np.where(overlap == max_overlap)

        fg_index = anchor_assignto_gt
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = gt_assignto_anchor

        # regression
        fg_index = np.where(label != 0)
        target_window = anchor_boxes[fg_index]
        target_truth_box = truth_boxes[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box)
        target_weight[fg_index] = 1

        # weights for class balancing
        fg_index = np.where((label_weight != 0) & (label != 0))[0]
        bg_index = np.where((label_weight != 0) & (label == 0))[0]

        num_fg = len(fg_index)
        num_bg = len(bg_index)
        label_weight[fg_index] = 1
        label_weight[bg_index] = num_fg / num_bg

        # task balancing
        target_weight[fg_index] = label_weight[fg_index]

    # save
    label = torch.from_numpy(label).to(cfg.device)
    label_assign = torch.from_numpy(label_assign).to(cfg.device)
    label_weight = torch.from_numpy(label_weight).to(cfg.device)
    target = torch.from_numpy(target).to(cfg.device)
    target_weight = torch.from_numpy(target_weight).to(cfg.device)
    return label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, anchor_boxes, truth_boxes_batch):
    """
    append -> concat -> to tensor
    :param cfg:
    :param images:
    :param anchor_boxes:
    :param truth_boxes_batch:
    :return:
    """
    truth_boxes_batch = truth_boxes_batch.detach().cpu().numpy() if type(truth_boxes_batch) is torch.Tensor else truth_boxes_batch
    
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
