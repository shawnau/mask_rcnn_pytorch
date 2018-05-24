import torch
import numpy as np
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rpn.rpn_nms import rpn_encode

class RPNTarget:
    def __init__(self, cfg):
        self.cfg = cfg

    def label_anchor_boxes(self, anchor_boxes, truth_boxes):
        """
        :param anchor_boxes:
        :param truth_boxes:
        :return:
             label: 1 for foreground, 0 for background
             label_assign: which truth box is assigned to the anchor
             label_weight: foreground or background=1, otherwise=0 (don't care)
        """
        num_anchor_boxes = len(anchor_boxes)
        num_truth_box = len(truth_boxes)

        label = np.zeros((num_anchor_boxes,), np.float32)
        label_assign = np.zeros((num_anchor_boxes,), np.int32)
        label_weight = np.zeros((num_anchor_boxes,), np.float32)  # why init with 1? todo: init with 0 ?

        overlap = cython_box_overlap(anchor_boxes, truth_boxes)
        argmax_overlap = np.argmax(overlap, 1)  # assign truth box's index to each anchor

        # label 1/0 for each anchor by threshold
        max_overlap = overlap[np.arange(num_anchor_boxes), argmax_overlap]
        bg_index = max_overlap < self.cfg.rpn_train_bg_thresh_high
        label[bg_index] = 0
        label_weight[bg_index] = 1

        fg_index = max_overlap >= self.cfg.rpn_train_fg_thresh_low
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap

        # each ground_truth box must be assigned
        # re-assign less overlapped gt to anchor_boxes, nomatter it's box_overlap
        argmax_overlap = np.argmax(overlap, 0)  # assign anchor's index to each truth box
        max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]
        anchor_assignto_gt, gt_assignto_anchor = np.where(overlap == max_overlap)

        fg_index = anchor_assignto_gt
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = gt_assignto_anchor

        return label, label_assign, label_weight

    def calculate_delta(self, anchor_boxes, truth_boxes, label, label_assign):
        """
        todo: remove redundant targets when returning ?
        :return:
            target: bboxes' offsets
            target_weight: same as label_weight
        """
        num_anchor_boxes = len(anchor_boxes)

        target = np.zeros((num_anchor_boxes, 4), np.float32)
        target_weight = np.zeros((num_anchor_boxes,), np.float32)

        # regression
        fg_index = np.where(label != 0)
        target_window = anchor_boxes[fg_index]
        target_truth_box = truth_boxes[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box)
        target_weight[fg_index] = 1

        return target, target_weight

    def balance(self, image, label, label_weight, target_weight):
        """
        1. weighting foreground and background for label and target
        2. weighting different scales
        :param image:
        :param label: 1 for foreground, 0 for background
        :param label_weight: foreground or background=1, otherwise=0 (don't care)
        :param target_weight: same as label_weight
        :return:
            label_weight: foreground=1, background = \in (0, 1] by rareness,
            target_weight: same as label_weight
        """
        # weights for class balancing
        fg_index = np.where((label_weight != 0) & (label != 0))[0]
        bg_index = np.where((label_weight != 0) & (label == 0))[0]

        num_fg = len(fg_index)
        num_bg = len(bg_index)
        label_weight[fg_index] = 1
        label_weight[bg_index] = num_fg / num_bg

        if self.cfg.rpn_train_scale_balance:
            # weights for scale balancing
            _, height, width = image.size()
            num_scales = len(self.cfg.rpn_scales)
            num_bases = [len(b) for b in self.cfg.rpn_base_apsect_ratios]
            start = 0
            for l in range(num_scales):
                h, w = int(height // 2 ** l), int(width // 2 ** l)
                end = start + h * w * num_bases[l]
                label_weight[start:end] *= (2 ** l) ** 2
                start = end

        # assign weight to target the same as label
        target_weight[fg_index] = label_weight[fg_index]
        return label_weight, target_weight

    def make_one_rpn_target(self, image, anchor_boxes, truth_boxes):
        label, label_assign, label_weight = self.label_anchor_boxes(anchor_boxes, truth_boxes)
        target, target_weight = self.calculate_delta(anchor_boxes, truth_boxes, label, label_assign)
        label_weight, target_weight = self.balance(image, label, label_weight, target_weight)

        label         = torch.from_numpy(label        ).to(self.cfg.device)
        label_assign  = torch.from_numpy(label_assign ).to(self.cfg.device)
        label_weight  = torch.from_numpy(label_weight ).to(self.cfg.device)
        target        = torch.from_numpy(target       ).to(self.cfg.device)
        target_weight = torch.from_numpy(target_weight).to(self.cfg.device)

        return label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, images, anchor_boxes, truth_boxes_batch):
    """
    append -> concat -> to tensor
    :param cfg:
    :param images:
    :param anchor_boxes:
    :param truth_boxes_batch:
    :return:
    """
    anchor_labels = []
    anchor_label_assigns = []
    anchor_label_weights = []
    anchor_targets = []
    anchor_targets_weights = []

    batch_size = len(truth_boxes_batch)
    for b in range(batch_size):
        image = images[b]
        truth_boxes = truth_boxes_batch[b]

        label, label_assign, label_weight, target, targets_weight = \
            RPNTarget(cfg).make_one_rpn_target(image, anchor_boxes, truth_boxes)

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
