import copy
import torch
import numpy as np
from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.utils.box_utils import is_small_box, resize_instance
from net.layer.rpn.rpn_utils import rpn_encode
from net.layer.rcnn.rcnn_utils import rcnn_encode


class MakeTarget:
    def __init__(self, cfg, mode, proposal, truth_box, truth_label=None, truth_instance=None):

        proposal = proposal.detach().numpy() if type(proposal) is torch.Tensor else proposal
        truth_box = truth_box.detach().numpy() if type(truth_box) is torch.Tensor else truth_box
        truth_label = truth_label.detach().numpy() if type(truth_label) is torch.Tensor else truth_label
        truth_instance = truth_instance.detach().numpy() if type(truth_instance) is torch.Tensor else truth_instance

        self.proposal = proposal if mode in ['rcnn', 'mask'] else None
        self.box = proposal if mode in ['rpn'] else self.proposal[:, 1:5]
        self.num_proposal = len(proposal)
        self.num_truth_box = len(truth_box)
        self.truth_box = truth_box
        self.truth_label = truth_label
        self.truth_instance = truth_instance
        # layer-specific configurations
        self.mode = mode
        self.device = cfg.device
        self.min_box_size = None
        self.mask_size = None
        self.fg_threshold_low = None
        self.bg_threshold_high = None
        self.bg_threshold_low = None
        self.resample_size = None
        self.fg_ratio = None
        self.init_mode(cfg, mode)
        # data to return
        self.fg_index = None
        self.bg_index = None
        self.box_label = None
        self.box_assign = None
        self.box_weight = None
        self.box_delta = None
        self.box_delta_weight = None
        self.instance = None

    def init_mode(self, cfg, mode):
        if mode in ['rpn']:
            self.fg_threshold_low = cfg.rpn_train_fg_thresh_low
            self.bg_threshold_high = cfg.rpn_train_bg_thresh_high
        elif mode in ['rcnn']:
            self.fg_threshold_low = cfg.rcnn_train_fg_thresh_low
            self.bg_threshold_high = cfg.rcnn_train_bg_thresh_high
            self.bg_threshold_low = cfg.rcnn_train_bg_thresh_low
            self.resample_size = cfg.rcnn_train_batch_size
            self.fg_ratio = cfg.rcnn_train_fg_fraction
            self.min_box_size = cfg.mask_train_min_size
        elif mode in ['mask']:
            self.fg_threshold_low = cfg.mask_train_fg_thresh_low
            self.resample_size = cfg.mask_train_batch_size
            self.fg_ratio = 1.0
            self.min_box_size = cfg.mask_train_min_size
            self.mask_size = cfg.mask_size

    def filter_box(self):
        valid_index = []
        for i in range(self.num_proposal):
            box = self.box[i]
            if not (is_small_box(box, min_size=self.min_box_size)):
                valid_index.append(i)

        self.proposal = self.proposal[valid_index]
        self.box = self.box[valid_index]
        self.num_proposal = len(valid_index)

    def label_box(self):
        self.box_label = np.zeros((self.num_proposal,), np.int64)
        self.box_assign = np.zeros((self.num_proposal,), np.int64)
        self.box_weight = np.zeros((self.num_proposal,), np.float32)
        if self.num_proposal <= 0 or self.num_truth_box <= 0: return
        overlap = cython_box_overlap(self.box, self.truth_box)
        argmax_overlap = np.argmax(overlap, 1)  # assign truth box's index to each anchor
        max_overlap = overlap[np.arange(self.num_proposal), argmax_overlap]

        if self.mode in ['rpn']:
            if self.num_proposal <= 0 or self.num_truth_box <= 0: return
            # label 1/0 for each anchor by threshold
            bg_index = max_overlap < self.bg_threshold_high
            self.box_label[bg_index] = 0
            self.box_weight[bg_index] = 1

            fg_index = max_overlap >= self.fg_threshold_low
            self.box_label[fg_index] = 1
            self.box_weight[fg_index] = 1
            self.box_assign[...] = argmax_overlap

            # each ground_truth box must be assigned
            # re-assign less overlapped gt to anchor_boxes, nomatter it's box_overlap
            argmax_overlap = np.argmax(overlap, 0)  # assign anchor's index to each truth box
            max_overlap = overlap[argmax_overlap, np.arange(self.num_truth_box)]
            proposal_to_truth, truth_to_proposal = np.where(overlap == max_overlap)

            fg_index = proposal_to_truth
            self.box_label[fg_index] = 1
            self.box_weight[fg_index] = 1
            self.box_assign[fg_index] = truth_to_proposal
            self.fg_index = np.where((self.box_weight != 0) & (self.box_label != 0))[0]
            self.bg_index = np.where((self.box_weight != 0) & (self.box_label == 0))[0]

            # re-weight by rareness
            num_fg = len(self.fg_index)
            num_bg = len(self.bg_index)
            self.box_weight[fg_index] = 1
            self.box_weight[bg_index] = num_fg / num_bg

        if self.mode in ['rcnn', 'mask']:
            if self.num_proposal <= 0 or self.num_truth_box <= 0: return
            self.fg_index = np.where(max_overlap >= self.fg_threshold_low)[0]
            if self.mode == 'rcnn':
                self.bg_index = np.where((max_overlap < self.bg_threshold_high) & \
                                         (max_overlap >= self.bg_threshold_low))[0]
            elif self.mode == 'mask':
                self.bg_index = np.array([])
            self.fg_index, self.bg_index = self.balance(self.fg_index, self.bg_index)
            self.subsample(argmax_overlap)

    def subsample(self, argmax_overlap):
        if self.mode in ['rcnn', 'mask']:
            subsample_index = np.concatenate([self.fg_index, self.bg_index], 0).astype(np.int64)  # int64+empty=float64
            self.proposal = self.proposal[subsample_index]
            self.box = self.box[subsample_index]
            self.box_assign = argmax_overlap[subsample_index]
            self.box_label = self.truth_label[self.box_assign]
            self.box_label[len(self.fg_index):] = 0  # clamp labels for the background to 0
            self.num_proposal = len(subsample_index)

    def box_regression(self):
        self.box_delta = np.zeros((self.num_proposal, 4), np.float32)
        self.box_delta_weight = np.zeros((self.num_proposal,), np.float32)
        if self.num_proposal <= 0 or self.num_truth_box <= 0: return

        if self.mode in ['rpn']:
            fg_box = self.box[self.fg_index]  # calculate foreground only
            truth_box = self.truth_box[self.box_assign[self.fg_index]]
            self.box_delta[self.fg_index] = rpn_encode(fg_box, truth_box)
            self.box_delta_weight[self.fg_index] = self.box_weight[self.fg_index]

        elif self.mode in ['rcnn']:
            truth_box = self.truth_box[self.box_assign[:len(self.fg_index)]]
            fg_box = self.box[:len(self.fg_index)]
            self.box_delta = rcnn_encode(fg_box, truth_box)

    def make_instance(self):
        self.instance = np.zeros((self.num_proposal, self.mask_size, self.mask_size), np.float32)
        if self.num_proposal <= 0 or self.num_truth_box <= 0: return
        resized_instance = []
        for i in range(len(self.fg_index)):
            instance = self.truth_instance[self.box_assign[i]]
            box = self.box[i]
            crop = resize_instance(instance, box, self.mask_size)
            resized_instance.append(crop[np.newaxis, :, :])
        resized_instance = np.vstack(resized_instance)
        self.instance = resized_instance

    def balance(self, fg_index, bg_index):
        """
        1. resample sample into given resample_size and foreground ratio
        2. balance foreground and background fraction
        will generate duplicated proposals if proposals < rcnn batch size
        :param fg_index:
        :param bg_index:
        :return:
        """
        num = self.resample_size
        num_fg = int(np.round(self.fg_ratio * self.resample_size))

        # Small modification to the original version where we ensure a fixed number of regions are sampled
        # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
        fg_length = len(fg_index)
        bg_length = len(bg_index)

        if fg_length > 0 and bg_length > 0:
            num_fg = min(num_fg, fg_length)
            fg_index = fg_index[
                np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
            ]
            num_bg = num - num_fg
            bg_index = bg_index[
                np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
            ]
        # no bgs
        elif fg_length > 0:
            num_fg = num
            num_bg = 0
            fg_index = fg_index[
                np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
            ]
        # no fgs
        elif bg_length > 0:
            num_fg = 0
            num_bg = num
            bg_index = bg_index[
                np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
            ]
        # no bgs and no fgs?
        else:
            num_fg = 0
            num_bg = num
            bg_index = np.random.choice(self.num_proposal, size=num_bg, replace=self.num_proposal < num_bg)

        assert ((num_fg + num_bg) == num)
        return fg_index, bg_index


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
    print(truth_box.shape)
    num_truth = len(truth_box)
    if num_truth != 0:
        truth = np.zeros((num_truth, 7), np.float32)
        truth[:, 0] = img_idx
        truth[:, 1:5] = truth_box
        truth[:, 5] = score
        truth[:, 6] = truth_label
    else:
        truth = np.zeros((0, 7), np.float32)
    total_proposals = np.vstack([proposal, truth])
    return total_proposals


def make_rpn_target(cfg, anchor_boxes, truth_box_batch):
    box_label_batch = []
    box_assign_batch = []
    box_label_weight_batch = []
    box_delta = []
    box_delta_weight = []

    for b in range(len(truth_box_batch)):
        truth_box = truth_box_batch[b]

        t = MakeTarget(cfg, 'rpn', anchor_boxes, truth_box)
        t.label_box()
        t.box_regression()

        label = torch.from_numpy(t.box_label).to(t.device)
        label_assign = torch.from_numpy(t.box_assign).to(t.device)
        label_weight = torch.from_numpy(t.box_weight).to(t.device)
        target = torch.from_numpy(t.box_delta).to(t.device)
        target_weight = torch.from_numpy(t.box_delta_weight).to(t.device)

        box_label_batch.append(label.view(1, -1))
        box_assign_batch.append(label_assign.view(1, -1))
        box_label_weight_batch.append(label_weight.view(1, -1))
        box_delta.append(target.view(1, -1, 4))
        box_delta_weight.append(target_weight.view(1, -1))

    box_label_batch = torch.cat(box_label_batch, 0)
    box_assign_batch = torch.cat(box_assign_batch, 0)
    box_label_weight_batch = torch.cat(box_label_weight_batch, 0)
    box_delta = torch.cat(box_delta, 0)
    box_delta_weight = torch.cat(box_delta_weight, 0)

    return box_label_batch, box_assign_batch, box_label_weight_batch, box_delta, box_delta_weight


def make_rcnn_target(cfg, proposal_batch, truth_box_batch, truth_label_batch):
    truth_box_batch = copy.deepcopy(truth_box_batch)
    truth_label_batch = copy.deepcopy(truth_label_batch)

    sampled_proposal = []
    box_label_batch = []
    box_assign_batch = []
    box_delta = []

    for b in range(len(truth_box_batch)):
        proposal = proposal_batch[proposal_batch[:, 0] == b]
        truth_box = truth_box_batch[b]
        truth_label = truth_label_batch[b]

        proposal = add_truth_box_to_proposal(proposal, b, truth_box, truth_label)

        t = MakeTarget(cfg, 'rcnn', proposal, truth_box, truth_label)
        t.filter_box()
        t.label_box()
        t.box_regression()

        proposal = torch.from_numpy(t.proposal).to(t.device)
        label = torch.from_numpy(t.box_label).to(t.device)
        label_assign = torch.from_numpy(t.box_assign).to(t.device)
        delta = torch.from_numpy(t.box_delta).to(t.device)

        sampled_proposal.append(proposal)
        box_label_batch.append(label)
        box_assign_batch.append(label_assign)
        box_delta.append(delta)

    sampled_proposal = torch.cat(sampled_proposal, 0)
    box_label_batch = torch.cat(box_label_batch, 0)
    box_assign_batch = np.hstack(box_assign_batch)
    box_delta = torch.cat(box_delta, 0)

    return sampled_proposal, box_label_batch, box_assign_batch, box_delta


def make_mask_target(cfg, proposal_batch, truth_box_batch, truth_label_batch, truth_instance_batch):
    truth_box_batch = copy.deepcopy(truth_box_batch)
    truth_label_batch = copy.deepcopy(truth_label_batch)
    truth_instance_batch = copy.deepcopy(truth_instance_batch)

    sampled_proposal = []
    box_label_batch = []
    sampled_truth_instance = []

    for b in range(len(truth_box_batch)):
        proposal = proposal_batch[proposal_batch[:, 0] == b]
        truth_box = truth_box_batch[b]
        truth_label = truth_label_batch[b]
        truth_instance = truth_instance_batch[b]

        proposal = add_truth_box_to_proposal(proposal, b, truth_box, truth_label)
        t = MakeTarget(cfg, 'mask', proposal, truth_box, truth_label, truth_instance)
        t.filter_box()
        t.label_box()
        t.make_instance()

        proposal = torch.from_numpy(t.proposal).to(t.device)
        label = torch.from_numpy(t.box_label).to(t.device)
        instance = torch.from_numpy(t.instance).to(t.device)

        sampled_proposal.append(proposal)
        box_label_batch.append(label)
        sampled_truth_instance.append(instance)

    sampled_proposal = torch.cat(sampled_proposal, 0)
    box_label_batch = torch.cat(box_label_batch, 0)
    sampled_truth_instance = torch.cat(sampled_truth_instance, 0)

    return sampled_proposal, box_label_batch, sampled_truth_instance
