import copy
import torch
import numpy as np

from net.utils.box_utils import is_small_box
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.layer.rcnn.rcnn_utils import rcnn_encode


def add_truth_box_to_proposal(proposal, img_idx, truth_box, truth_label, score=-1):
    """
    :param proposal: proposals fot ONE IMAGE. e.g.
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


def make_one_rcnn_target(cfg, proposals, truth_boxes, truth_labels):
    def balance(_fg_index, _bg_index):
        """
        balance foreground and background fraction
        will generate duplicated proposals if proposals < rcnn batch size
        :param _fg_index:
        :param _bg_index:
        :return:
        """
        num = cfg.rcnn_train_batch_size
        num_fg = int(np.round(cfg.rcnn_train_fg_fraction * cfg.rcnn_train_batch_size))

        # Small modification to the original version where we ensure a fixed number of regions are sampled
        # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
        fg_length = len(_fg_index)
        bg_length = len(_bg_index)

        if fg_length > 0 and bg_length > 0:
            num_fg = min(num_fg, fg_length)
            _fg_index = _fg_index[
                np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
            ]
            num_bg = num - num_fg
            _bg_index = _bg_index[
                np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
            ]
        # no bgs
        elif fg_length > 0:
            num_fg = num
            num_bg = 0
            _fg_index = _fg_index[
                np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
            ]
        # no fgs
        elif bg_length > 0:
            num_fg = 0
            num_bg = num
            _bg_index = _bg_index[
                np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
            ]
        # no bgs and no fgs?
        else:
            num_fg = 0
            num_bg = num
            _bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal < num_bg)

        assert ((num_fg + num_bg) == num)
        return _fg_index, _bg_index, num_fg

    sampled_proposal = torch.FloatTensor((0, 7) ).to(cfg.device)
    sampled_label    = torch.LongTensor((0, 1)  ).to(cfg.device)
    sampled_assign   = np.array((0, 1), np.int32)
    sampled_target   = torch.FloatTensor((0, 4) ).to(cfg.device)

    # filter invalid proposals
    num_proposal = len(proposals)

    valid = []
    for i in range(num_proposal):
        box = proposals[i, 1:5]
        if not (is_small_box(box, min_size=cfg.mask_train_min_size)):
            valid.append(i)

    proposals = proposals[valid]
    # assign fg/bg to each box
    num_proposal = len(proposals)
    box = proposals[:, 1:5]
    # for each bbox, the index of gt which has max box_overlap with it
    overlap = cython_box_overlap(box, truth_boxes)
    argmax_overlap = np.argmax(overlap, 1)
    max_overlap = overlap[np.arange(num_proposal), argmax_overlap]

    fg_index = np.where(max_overlap >= cfg.rcnn_train_fg_thresh_low)[0]
    bg_index = np.where((max_overlap < cfg.rcnn_train_bg_thresh_high) & \
                        (max_overlap >= cfg.rcnn_train_bg_thresh_low))[0]

    if len(truth_boxes) == 0 or len(proposals) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    fg_index, bg_index, num_fg = balance(fg_index, bg_index)

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
    
    if type(sampled_target) is torch.Tensor:
        print('tensor before convert: ', sampled_target.size()) 
    else: 
        print('ndarray before convert: ', sampled_target.shape)

    sampled_target   = sampled_target if type(sampled_target) is torch.Tensor else torch.from_numpy(sampled_target  ).to(cfg.device)
    print('converted tensor: ', sampled_target.size())
    sampled_label    = sampled_label if type(sampled_label) is torch.Tensor else torch.from_numpy(sampled_label   ).to(cfg.device)
    sampled_proposal = sampled_proposal if type(sampled_proposal) is torch.Tensor else torch.from_numpy(sampled_proposal).to(cfg.device)
    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, proposals, truth_boxes, truth_labels):
    """
    a sampled subset of proposals, with it's_train corresponding truth label and offsets
    :param images: (B, 3, H, W), BGR mode
    :param proposals: (B, 7), [i, x0, y0, x1, y1, score, label]
    :param truth_boxes: (B, _, 4)
    :param truth_labels: (B, _, 1)
    :return:
    """

    proposals = proposals.detach().cpu().numpy() if type(proposals) is torch.Tensor else proposals
    truth_boxes = truth_boxes.detach().cpu().numpy() if type(truth_boxes) is torch.Tensor else truth_boxes
    truth_labels = truth_labels.detach().cpu().numpy() if type(truth_labels) is torch.Tensor else truth_labels

    # <todo> take care of don't care ground truth. Here, we only ignore them  ----
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)
    batch_size = len(truth_boxes)
    # filter truth labels is 0 todo: do we really need to check this?
    for img_idx in range(batch_size):
        index = np.where(truth_labels[img_idx] > 0)[0]
        truth_boxes[img_idx] = truth_boxes[img_idx][index]
        truth_labels[img_idx] = truth_labels[img_idx][index]

    # proposals = proposals # todo: proposals is good to go using tensor instead of ndarray
    sampled_proposals = []
    sampled_labels =    []
    sampled_assigns =   []
    sampled_targets =   []

    batch_size = len(truth_boxes)
    for img_idx in range(batch_size):
        img_truth_boxes  = truth_boxes[img_idx]
        img_truth_labels = truth_labels[img_idx]

        if len(img_truth_boxes) != 0:
            if len(proposals) == 0:
                img_proposals = np.zeros((0, 7), np.float32)
            else:
                img_proposals = proposals[proposals[:, 0] == img_idx]

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