import torch
from torch import nn

from net.layer.backbone.SE_ResNeXt_FPN import SEResNeXtFPN
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead
from net.layer.roi_align import RoiAlign

from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes, rpn_cls_loss, rpn_reg_loss
from net.layer.rcnn.rcnn_utils import rcnn_cls_loss, rcnn_reg_loss
from net.layer.mask.mask_utils import make_empty_masks, mask_loss

from net.layer.rpn.rpn_target import make_rpn_target
from net.layer.rcnn.rcnn_target import make_rcnn_target
from net.layer.mask.mask_target import make_mask_target

from net.layer.nms import rpn_nms, rcnn_nms, mask_nms


class RPN(nn.Module):
    def __init__(self, cfg, mode):
        super(RPN, self).__init__()
        self.cfg = cfg
        self.mode = mode
        feature_channels = 256
        self.rpn_head = RpnMultiHead(cfg, feature_channels)

    def forward(self, images, features):
        self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn_head(features)
        self.anchor_boxes = rpn_make_anchor_boxes(features, self.cfg)
        self.rpn_proposals = rpn_nms(self.cfg, self.mode, images,
                                     self.anchor_boxes,
                                     self.rpn_logits_flat,
                                     self.rpn_deltas_flat)
        return self.rpn_proposals

    def loss(self, images, features, truth_boxes):
        self.forward(images, features)

        rpn_labels, \
        rpn_label_assigns, \
        rpn_label_weights, \
        rpn_targets, \
        rpn_targets_weights = \
            make_rpn_target(self.cfg, self.anchor_boxes, truth_boxes)

        self.cls_loss = rpn_cls_loss(self.rpn_logits_flat,
                                     rpn_labels,
                                     rpn_label_weights)

        self.reg_loss = rpn_reg_loss(rpn_labels,
                                     self.rpn_deltas_flat,
                                     rpn_targets,
                                     rpn_targets_weights)

        return self.cls_loss, self.reg_loss


class RCNN(nn.Module):
    def __init__(self, cfg, mode):
        super(RCNN, self).__init__()
        self.cfg = cfg
        self.mode = mode
        crop_channels = 256
        self.rcnn_crop = RoiAlign(cfg, cfg.rcnn_crop_size)
        self.rcnn_head = RcnnHead(cfg, crop_channels)

        self.cls_loss = None
        self.reg_loss = None

    def forward(self, images, features, rpn_proposals):
        rcnn_crops = self.rcnn_crop(features, self.rpn_proposals)
        self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
        self.rcnn_proposals = rcnn_nms(self.cfg, self.mode, images,
                                       rpn_proposals,
                                       self.rcnn_logits,
                                       self.rcnn_deltas)
        return self.rcnn_proposals

    def loss(self, images, features, rpn_proposals, truth_boxes, truth_labels):
        sampled_rcnn_proposals, \
        sampled_rcnn_labels, \
        sampled_rcnn_assigns, \
        sampled_rcnn_targets = \
            make_rcnn_target(self.cfg, images, rpn_proposals, truth_boxes, truth_labels)

        if len(sampled_rcnn_proposals) > 0:
            self.forward(images, features, sampled_rcnn_proposals)

            self.cls_loss = rcnn_cls_loss(self.rcnn_logits, sampled_rcnn_labels)
            self.reg_loss = rcnn_reg_loss(sampled_rcnn_labels, self.rcnn_deltas, sampled_rcnn_targets)

        return self.cls_loss, self.reg_loss


class MaskNet(nn.Module):
    def __init__(self, cfg):
        super(MaskNet, self).__init__()
        self.cfg = cfg
        crop_channels = 256
        self.mask_crop = RoiAlign(cfg, cfg.mask_crop_size)
        self.mask_head = MaskHead(cfg, crop_channels)

        self.mask_cls_loss = None

    def forward(self, images, features, rcnn_proposals):
        mask_crops = self.mask_crop(features, self.detections)
        self.mask_logits = self.mask_head(mask_crops)
        self.masks, self.mask_instances, self.mask_proposals = \
            mask_nms(self.cfg, images, rcnn_proposals, self.mask_logits)

        return self.masks, self.mask_instances, self.mask_proposals

    def loss(self, images, features, rcnn_proposals, truth_boxes, truth_labels, truth_instances):
        sampled_rcnn_proposals, \
        sampled_mask_labels, \
        sampled_mask_instances, = \
            make_mask_target(self.cfg, images, rcnn_proposals, truth_boxes, truth_labels, truth_instances)

        if len(sampled_rcnn_proposals) > 0:
            self.forward(images, features, sampled_rcnn_proposals)
            self.mask_cls_loss = mask_loss(self.mask_logits, sampled_mask_labels, sampled_mask_instances)

        return self.mask_cls_loss


class MaskRcnnNet(nn.Module):
    def __init__(self, cfg):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-se-resnext50-fpn\''
        self.cfg  = cfg
        self.mode = 'train_all'

        self.feature_net = SEResNeXtFPN([3, 4, 6, 3])
        self.rpn = RPN(cfg, self.mode)
        self.rcnn = RCNN(cfg, self.mode)
        self.mask_net = MaskNet(cfg)

        self.rpn_proposals = []
        self.rcnn_proposals = []
        self.masks = None
        self.mask_instances = None
        self.mask_proposals = None

        self.rpn_cls_loss  = None
        self.rpn_reg_loss  = None
        self.rcnn_cls_loss = None
        self.rcnn_reg_loss = None
        self.mask_cls_loss = None

    def forward(self, images):
        features = self.feature_net(images)
        self.rpn_proposals = self.rpn(images, features)
        self.rcnn_proposals = self.rcnn(images, features, self.rpn_proposals)
        self.masks, self.mask_instances, self.mask_proposals = self.mask_net(images, features, self.rcnn_proposals)

    def train_rpn(self, images, truth_boxes):
        features = self.feature_net(images)
        self.rpn_cls_loss, self.rpn_reg_loss = self.rpn.loss(images, features, truth_boxes)

    def train_rcnn(self, images, truth_boxes, truth_labels):
        features = self.feature_net(images)
        self.rpn_cls_loss, self.rpn_reg_loss = self.rpn.loss(images, features, truth_boxes)
        self.rcnn_cls_loss, self.rcnn_reg_loss = self.rcnn.loss(images, features, self.rpn.rpn_proposals,
                                                                truth_boxes, truth_labels)

    def train_all(self, images, truth_boxes, truth_labels, truth_instances):
        features = self.feature_net(images)
        self.rpn_cls_loss, self.rpn_reg_loss = self.rpn.loss(images, features, truth_boxes)
        self.rcnn_cls_loss, self.rcnn_reg_loss = self.rcnn.loss(images, features, self.rpn.rpn_proposals,
                                                                truth_boxes, truth_labels)
        self.mask_cls_loss = self.mask_net.loss(images, features, self.rcnn.rcnn_proposals,
                                                truth_boxes, truth_labels, truth_instances)

    def loss(self, images, truth_boxes, truth_labels, truth_instances):
        if self.mode in ['train_rpn', 'valid_rpn']:
            self.train_rpn(images, truth_boxes)
            total_loss = self.rpn_cls_loss + self.rpn_reg_loss
        elif self.mode in ['train_rcnn', 'valid_rcnn']:
            self.train_rcnn(images, truth_boxes, truth_labels)
            total_loss = self.rpn_cls_loss + self.rpn_reg_loss + \
                         self.rcnn_cls_loss + self.rcnn_reg_loss
        elif self.mode in ['train_all', 'valid_all']:
            self.train_all(images, truth_boxes, truth_labels, truth_instances)
            total_loss = self.rpn_cls_loss + self.rpn_reg_loss + \
                         self.rcnn_cls_loss + self.rcnn_reg_loss + self.mask_cls_loss
        else:
            raise KeyError('mode %s note recognized' % self.mode)

        return total_loss

    def set_mode(self, mode):
        self.mode = mode
        self.rpn.mode = mode
        self.rcnn.mode = mode
        if mode in ['eval', 'valid_rpn', 'valid_rcnn', 'valid_all', 'test']:
            self.eval()
        elif mode in ['train_rpn', 'train_rcnn', 'train_all']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)