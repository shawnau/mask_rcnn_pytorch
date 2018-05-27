import torch
from torch import nn

from net.layer.backbone.SE_ResNeXt_FPN import SEResNeXtFPN
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead
from net.layer.roi_align.crop import CropRoi

from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes, rpn_cls_loss, rpn_reg_loss
from net.layer.rcnn.rcnn_utils import rcnn_cls_loss, rcnn_reg_loss
from net.layer.mask.mask_utils import make_empty_masks, mask_loss

from net.layer.rpn.rpn_target import make_rpn_target
from net.layer.rcnn.rcnn_target import make_rcnn_target
from net.layer.mask.mask_target import make_mask_target

from net.layer.nms import rpn_nms, rcnn_nms, mask_nms


class MaskRcnnNet(nn.Module):

    def __init__(self, cfg):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-se-resnext50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels = 256
        crop_channels = feature_channels
        self.feature_net = SEResNeXtFPN([3, 4, 6, 3])
        self.rpn_head    = RpnMultiHead(cfg, feature_channels)
        self.rcnn_crop   = CropRoi  (cfg, cfg.rcnn_crop_size)
        self.rcnn_head   = RcnnHead (cfg, crop_channels)
        self.mask_crop   = CropRoi  (cfg, cfg.mask_crop_size)
        self.mask_head   = MaskHead (cfg, crop_channels)

    def forward(self, images, truth_boxes=None, truth_labels=None, truth_instances=None):
        features = self.feature_net(images)

        # rpn proposals -------------------------------------------
        self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn_head(features)
        self.anchor_boxes = rpn_make_anchor_boxes(features, self.cfg)
        self.rpn_proposals = rpn_nms(self.cfg, self.mode, images,
                                     self.anchor_boxes,
                                     self.rpn_logits_flat,
                                     self.rpn_deltas_flat)

        # make tagets for rpn and rcnn ------------------------------------------------
        if self.mode in ['train', 'valid']:
            self.rpn_labels, \
            self.rpn_label_assigns, \
            self.rpn_label_weights, \
            self.rpn_targets, \
            self.rpn_targets_weights = \
                make_rpn_target(self.cfg, self.anchor_boxes, truth_boxes)

            if len(self.rpn_proposals) > 0:
                self.sampled_rcnn_proposals, \
                self.sampled_rcnn_labels, \
                self.sampled_rcnn_assigns, \
                self.sampled_rcnn_targets = \
                    make_rcnn_target(self.cfg, images, self.rpn_proposals, truth_boxes, truth_labels)

                self.rpn_proposals = self.sampled_rcnn_proposals  # use sampled proposals for training

        # rcnn proposals ------------------------------------------------
        if len(self.rpn_proposals) > 0:
            rcnn_crops = self.rcnn_crop(features, self.rpn_proposals)
            self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
            self.rcnn_proposals = rcnn_nms(self.cfg, self.mode, images,
                                           self.rpn_proposals,
                                           self.rcnn_logits,
                                           self.rcnn_deltas)
        else:
            self.rcnn_proposals = self.rpn_proposals  # for eval only when no rpn proposals

        # make targets for mask head ------------------------------------
        if self.mode in ['train', 'valid'] and len(self.rcnn_proposals) > 0:
            self.sampled_rcnn_proposals, \
            self.sampled_mask_labels, \
            self.sampled_mask_instances,   = \
                make_mask_target(self.cfg,
                                 images,
                                 self.rcnn_proposals,
                                 truth_boxes,
                                 truth_labels,
                                 truth_instances)
            self.rcnn_proposals = self.sampled_rcnn_proposals

        # segmentation  -------------------------------------------
        self.detections = self.rcnn_proposals
        self.masks = make_empty_masks(self.cfg, self.mode, images)
        self.mask_instances = []

        if len(self.rcnn_proposals) > 0:
            mask_crops = self.mask_crop(features, self.detections)
            self.mask_logits = self.mask_head(mask_crops)
            self.masks, self.mask_instances, self.mask_proposals = \
                mask_nms(self.cfg, images, self.rcnn_proposals, self.mask_logits)
            self.detections = self.mask_proposals

    def loss(self):
        self.rpn_cls_loss = rpn_cls_loss(self.rpn_logits_flat,
                                         self.rpn_labels,
                                         self.rpn_label_weights)

        self.rpn_reg_loss = rpn_reg_loss(self.rpn_labels,
                                         self.rpn_deltas_flat,
                                         self.rpn_targets,
                                         self.rpn_targets_weights)
        if len(self.rcnn_proposals) > 0:
            self.rcnn_cls_loss = rcnn_cls_loss(self.rcnn_logits,
                                               self.sampled_rcnn_labels)

            self.rcnn_reg_loss = rcnn_reg_loss(self.sampled_rcnn_labels,
                                               self.rcnn_deltas,
                                               self.sampled_rcnn_targets)

            self.mask_cls_loss  = mask_loss(self.mask_logits,
                                            self.sampled_mask_labels,
                                            self.sampled_mask_instances)

            self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss + \
                              self.rcnn_cls_loss + self.rcnn_reg_loss + \
                              self.mask_cls_loss
        else:
            self.rcnn_cls_loss = torch.tensor(0.0)
            self.rcnn_reg_loss = torch.tensor(0.0)
            self.mask_cls_loss = torch.tensor(0.0)
            self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss

        return self.total_loss

    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)