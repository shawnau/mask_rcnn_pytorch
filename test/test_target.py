import unittest
from torch.utils.data import DataLoader
from loader.dsb2018.dataset import *
from loader.sampler import *

from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes, rpn_cls_loss, rpn_reg_loss

from net.layer.rpn.rpn_target import make_rpn_target
from net.layer.rcnn.rcnn_target import make_rcnn_target
from net.layer.mask.mask_target import make_mask_target

from configuration import Configuration

from net.layer.rcnn.rcnn_utils import rcnn_reg_loss, rcnn_cls_loss
from net.layer.mask.mask_utils import mask_loss


class TestTarget(unittest.TestCase):
    """
    img_size = 256*256
    truth_boxes = \
        [[ 53., 162.,  69., 177.],
         [149., 190., 179., 213.],
         [125.,  32., 144.,  49.],
         [247., 174., 255., 190.],
         [ 41.,  93.,  63., 112.],
         [167., 101., 189., 118.]]
    truth_labels = \
        [1., 1., 1., 1., 1., 1.]

    """
    def setUp(self):
        self.cfg = Configuration()
        self.cfg.batch_size = 1
        # loader
        train_dataset = ScienceDataset(self.cfg, 'test', mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate)
        # backbone features
        p2 = torch.randn(self.cfg.batch_size, 256, 128, 128)
        p3 = torch.randn(self.cfg.batch_size, 256, 64, 64)
        p4 = torch.randn(self.cfg.batch_size, 256, 32, 32)
        p5 = torch.randn(self.cfg.batch_size, 256, 16, 16)
        self.fs = [p2, p3, p4, p5]
        # rpn_proposals
        rpn_proposals = np.array(
            [[0., 52., 161., 70., 176., 0.6, 1.],
             [0., 148., 190., 176., 215., 0.7, 1.],
             [0., 130., 30., 150., 60., 0.5, 1.],
             [0., 245., 170., 254., 200., 0.7, 1.],
             [0., 40., 90., 63., 100., 0.8, 1.],
             [0., 170., 120., 189., 128., 0.9, 1.]]
        ).astype(np.float32)  # overlap must use float32
        self.rpn_proposals = torch.from_numpy(rpn_proposals).to(self.cfg.device)
        self.rcnn_proposals = self.rpn_proposals

        self.num_anchor = \
            len(self.cfg.rpn_base_aspect_ratios[0]) * 128 * 128 + \
            len(self.cfg.rpn_base_aspect_ratios[1]) * 64 * 64 + \
            len(self.cfg.rpn_base_aspect_ratios[2]) * 32 * 32 + \
            len(self.cfg.rpn_base_aspect_ratios[3]) * 16 * 16

    def test_rpn(self):
        print('=' * 10, 'Test RPN Target', '=' * 10)
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            inputs: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """
            # truth_boxes = [np.zeros((0, 4), np.float32) for _ in range(self.cfg.batch_size)]  # let's fake a box
            anchor_boxes = rpn_make_anchor_boxes(self.fs, self.cfg)

            anchor_labels, \
            anchor_label_assigns, \
            anchor_label_weights, \
            anchor_targets, \
            anchor_targets_weights = make_rpn_target(self.cfg, anchor_boxes, truth_boxes)

            # test loss
            rpn_logits_flat = torch.randn(self.cfg.batch_size, self.num_anchor, 2)
            cls_loss = rpn_cls_loss(rpn_logits_flat,
                                    anchor_labels,
                                    anchor_label_weights)
            print('cls_loss: ', cls_loss)

            rpn_deltas_flat = torch.randn(self.cfg.batch_size, self.num_anchor, 2, 4)
            reg_loss = rpn_reg_loss(anchor_labels,
                                    rpn_deltas_flat,
                                    anchor_targets,
                                    anchor_targets_weights)
            print('reg_loss: ', reg_loss)

            anchor_labels = anchor_labels.numpy()
            anchor_label_assigns = anchor_label_assigns.numpy()
            anchor_label_weights = anchor_label_weights.numpy()
            anchor_targets = anchor_targets.numpy()
            anchor_targets_weights = anchor_targets_weights.numpy()

            print('anchor_labels: ',          anchor_labels.shape,        'positive samples: ', anchor_labels.sum())
            print('anchor_label_assigns: ',   anchor_label_assigns.shape, 'all samples: ',      len(anchor_label_assigns[(anchor_label_assigns > 0)]))
            print('anchor_label_weights: ',   anchor_label_weights.shape, 'label weight > 0: ', len(anchor_label_weights[anchor_label_weights > 0]))
            print('anchor_targets: ',         anchor_targets.shape)
            print('anchor_targets_weights: ', anchor_targets_weights.shape, 'target weight > 0: ', len(anchor_targets_weights[anchor_targets_weights > 0]))

    def test_rcnn(self):
        print('=' * 10, 'Test RCNN Target', '=' * 10)

        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            #truth_boxes = [np.zeros((0, 4), np.float32)]  # let's fake an empty box
            #rpn_proposals = torch.Tensor([])
            sampled_proposals, \
            sampled_labels, \
            sampled_assigns, \
            sampled_targets = make_rcnn_target(self.cfg, inputs, self.rpn_proposals, truth_boxes, truth_labels)

            print('sampled_proposals: ', sampled_proposals.size())
            print('sampled_labels: ',    sampled_labels)
            print('sampled_assigns: ',   sampled_assigns)
            print('sampled_targets: ',   sampled_targets)

            # test loss
            rcnn_logits = torch.randn(self.cfg.rcnn_train_batch_size, 2)
            rcnn_deltas = torch.randn(self.cfg.rcnn_train_batch_size, 8)

            cls_loss = rcnn_cls_loss(rcnn_logits, sampled_labels)
            reg_loss = rcnn_reg_loss(sampled_labels, rcnn_deltas, sampled_targets)
            print('cls_loss: ', cls_loss)
            print('reg_loss: ', reg_loss)

    def test_mask(self):
        print('=' * 10, 'Test MASK Target', '=' * 10)

        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            sampled_proposals, \
            sampled_labels, \
            sampled_instances = make_mask_target(self.cfg, inputs, self.rcnn_proposals, truth_boxes, truth_labels, truth_instances)
            print('sampled_proposals: ', sampled_proposals.size())
            print('sampled_labels: ', sampled_labels.size())
            print('sampled_instances: ', sampled_instances.size())

            # test_loss
            mask_logits = torch.randn(self.cfg.rcnn_train_batch_size,
                                      self.cfg.num_classes,
                                      2 * self.cfg.mask_crop_size,
                                      2 * self.cfg.mask_crop_size)
            cls_loss = mask_loss(mask_logits,
                                 sampled_labels,
                                 sampled_instances)
            print('mask loss: ', cls_loss)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestTarget('test_rpn'),
        TestTarget('test_rcnn'),
        TestTarget('test_mask')
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)