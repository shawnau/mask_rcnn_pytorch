import copy
import torch
import numpy as np
from net.utils.func_utils import to_tensor
from net.utils.box_utils import is_small_box
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rcnn.rcnn_utils import rcnn_encode


def add_truth_box_to_proposal(proposal, img_idx, truth_box, truth_label, score=-1):
    """
    :param proposal: rpn_proposals fot ONE IMAGE. e.g.
        [i, x0, y0, x1, y1, score, label]
    :param img_idx: image index in the batch
    :param truth_box:
    :param truth_label:
    :param score:
    :return:
    """
    if len(truth_box) != 0:
        truth = np.zeros((len(truth_box), 7), np.float32)
        truth[:, 0] = img_idx
        truth[:, 1:5] = truth_box
        truth[:, 5] = score
        truth[:, 6] = truth_label
    else:
        truth = np.zeros((0, 7), np.float32)
    total_proposals = np.vstack([proposal, truth])
    return total_proposals


def balance(fg_index, bg_index, batch_size, fg_ratio, num_proposal):
    """
    balance foreground and background fraction
    will generate duplicated rpn_proposals if rpn_proposals < rcnn batch size
    :param fg_index: foreground indices
    :param bg_index: background indices
    :param batch_size: rcnn train batch size
    :param fg_ratio: rcnn train foreground ratio
    :param num_proposal: number of rcnn rpn_proposals
    :return:
    """
    num_fg = int(np.round(fg_ratio * batch_size))
    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
    fg_length = len(fg_index)
    bg_length = len(bg_index)

    if fg_length > 0 and bg_length > 0:
        num_fg = min(num_fg, fg_length)
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
        ]
        num_bg = batch_size - num_fg
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
        ]
    # no bgs
    elif fg_length > 0:
        num_fg = batch_size
        num_bg = 0
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
        ]
    # no fgs
    elif bg_length > 0:
        num_fg = 0
        num_bg = batch_size
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
        ]
    # no bgs and no fgs?
    else:
        num_fg = 0
        num_bg = batch_size
        bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal < num_bg)

    assert ((num_fg + num_bg) == batch_size)
    return fg_index, bg_index, num_fg


def make_one_rcnn_target(cfg, proposals, truth_boxes, truth_labels):
    sampled_proposal = torch.zeros((0, 7), dtype=torch.float32).to(cfg.device)
    sampled_label    = torch.zeros((0, ), dtype=torch.int64).to(cfg.device)
    sampled_assign   = np.zeros((0, ), np.int32)
    sampled_target   = torch.zeros((0, 4), dtype=torch.float32).to(cfg.device)

    # filter invalid rpn_proposals
    num_proposal = len(proposals)
    valid = []
    for i in range(num_proposal):
        box = proposals[i, 1:5]
        if not (is_small_box(box, min_size=cfg.mask_train_min_size)):
            valid.append(i)
    proposals = proposals[valid]

    # assign fg/bg to each box
    num_proposal = len(proposals)
    if len(truth_boxes) > 0 and num_proposal > 0:
        box = proposals[:, 1:5]
        # for each bbox, the index of gt which has max box_overlap with it
        overlap = cython_box_overlap(box, truth_boxes)
        argmax_overlap = np.argmax(overlap, 1)
        max_overlap = overlap[np.arange(num_proposal), argmax_overlap]

        fg_index = np.where(max_overlap >= cfg.rcnn_train_fg_thresh_low)[0]
        bg_index = np.where((max_overlap < cfg.rcnn_train_bg_thresh_high) & \
                            (max_overlap >= cfg.rcnn_train_bg_thresh_low))[0]

        fg_index, bg_index, num_fg = balance(fg_index, bg_index,
                                             cfg.rcnn_train_batch_size,
                                             cfg.rcnn_train_fg_fraction,
                                             num_proposal)

        # selecting both fg and bg
        fg_bg_index = np.concatenate([fg_index, bg_index], 0)
        sampled_proposal = proposals[fg_bg_index]

        # label
        sampled_assign = argmax_overlap[fg_bg_index]
        sampled_label = truth_labels[sampled_assign]
        sampled_label[num_fg:] = 0  # Clamp labels for the background to 0

        # target
        if num_fg > 0:
            target_truth_box = truth_boxes[sampled_assign[:num_fg]]
            target_box = sampled_proposal[:num_fg][:, 1:5]
            sampled_target = rcnn_encode(target_box, target_truth_box)

        sampled_target   = to_tensor(sampled_target, cfg.device)
        sampled_label    = to_tensor(sampled_label, cfg.device)
        sampled_proposal = to_tensor(sampled_proposal, cfg.device)

    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, images, rpn_proposals, truth_boxes, truth_labels):
    """
    a sampled subset of rpn_proposals, with it's_train corresponding truth label and offsets
    :param images: (B, 3, H, W), BGR mode
    :param rpn_proposals_np: (B, 7), [i, x0, y0, x1, y1, score, label], B > 0
    :param truth_boxes: (B, _, 4)
    :param truth_labels: (B, _, 1)
    :return:
    """
    rpn_proposals_np = rpn_proposals.detach().cpu().numpy()
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)

    sampled_proposals = []
    sampled_labels =    []
    sampled_assigns =   []
    sampled_targets =   []

    for img_idx in range(len(images)):
        img_truth_boxes  = truth_boxes[img_idx]
        img_truth_labels = truth_labels[img_idx]

        if len(rpn_proposals_np) == 0:
            rpn_proposals_np = np.zeros((0, 7), np.float32)
        else:
            rpn_proposals_np = rpn_proposals_np[rpn_proposals_np[:, 0] == img_idx]

        img_proposals = rpn_proposals_np[rpn_proposals_np[:, 0] == img_idx]
        img_proposals = add_truth_box_to_proposal(img_proposals, img_idx, img_truth_boxes, img_truth_labels)

        sampled_proposal, sampled_label, sampled_assign, sampled_target = \
            make_one_rcnn_target(cfg, img_proposals, img_truth_boxes, img_truth_labels)

        sampled_proposals.append(sampled_proposal)
        sampled_labels.append(sampled_label)
        sampled_assigns.append(sampled_assign)
        sampled_targets.append(sampled_target)

    sampled_proposals = torch.cat(sampled_proposals, 0)
    sampled_labels    = torch.cat(sampled_labels, 0)
    sampled_targets   = torch.cat(sampled_targets, 0)
    sampled_assigns   = np.hstack(sampled_assigns)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_targets
