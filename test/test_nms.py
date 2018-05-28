import os
import unittest
from torch.utils.data import DataLoader
from loader.dsb2018.train_utils import *

from configuration import Configuration
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead
from net.layer.roi_align import RoiAlign

from net.layer.nms import rpn_nms, rcnn_nms, mask_nms


class TestNms(unittest.TestCase):
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
        self.cfg.data_dir = os.path.join(os.getcwd(), 'data')
        # loader
        split_file = os.path.join(os.getcwd(), 'data', 'test1')
        train_dataset = ScienceDataset(self.cfg, split_file, mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=make_collate)
        # backbone features
        p2 = torch.randn(5, 256, 128, 128)
        p3 = torch.randn(5, 256, 64, 64)
        p4 = torch.randn(5, 256, 32, 32)
        p5 = torch.randn(5, 256, 16, 16)
        self.fs = [p2, p3, p4, p5]
        # proposals
        rpn_proposals = np.array(
            [[0., 52.,  161., 70.,  176., 0.6, 1.],
             [0., 148., 190., 176., 215., 0.7, 1.],
             [0., 130., 30.,  150., 60.,  0.5, 1.],
             [0., 245., 170., 254., 200., 0.7, 1.],
             [0., 40.,  90.,  63.,  100., 0.8, 1.],
             [0., 170., 120., 189., 128., 0.9, 1.]]
        ).astype(np.float32)  # overlap must use float32
        self.rpn_proposals = torch.from_numpy(rpn_proposals).to(self.cfg.device)
        self.rcnn_proposals = self.rpn_proposals

    def test_rpn(self):
        net = RpnMultiHead(self.cfg, 256)
        logits_flat, deltas_flat = net(self.fs)

        print('=' * 10, 'Test RPN nms_func', '=' * 10)
        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            images: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """

            anchor_boxes = rpn_make_anchor_boxes(self.fs, self.cfg)
            rpn_proposals = rpn_nms(self.cfg, 'train', images, anchor_boxes, logits_flat, deltas_flat)
            print('rpn_proposals: ', rpn_proposals.size())

    def test_rcnn(self):
        print('=' * 10, 'Test RCNN nms_func', '=' * 10)
        crop = torch.randn(6, 256, 14, 14)  # 6 rpn_proposals got from rpn nms_func
        net = RcnnHead(self.cfg, 256)
        logits, deltas = net(crop)

        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            images: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """
            rcnn_proposals = rcnn_nms(self.cfg, 'train', images, self.rpn_proposals, logits, deltas)
            print('rcnn_proposals: ', rcnn_proposals.size())

    def test_empty_rcnn(self):
        roi_align = RoiAlign(self.cfg, self.cfg.rcnn_crop_size)

        empty_proposal = torch.zeros((1, 7))
        crop = roi_align(self.fs, empty_proposal)

        rcnn_head = RcnnHead(self.cfg, 256)
        logits, deltas = rcnn_head(crop)

        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            rcnn_proposals = rcnn_nms(self.cfg, 'train', images, empty_proposal, logits, deltas)
            print('rcnn_proposals for empty: ', rcnn_proposals)

    def test_mask(self):
        print('=' * 10, 'Test Mask nms_func', '=' * 10)

        crop = torch.randn(6, 256, 14, 14)  # 6 rcnn_proposals got from rpn nms_func
        net = MaskHead(self.cfg, 256)
        logits = net(crop)

        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            images: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """
            b_multi_masks, b_mask_instances, b_mask_proposals = mask_nms(self.cfg, images, self.rcnn_proposals, logits)
            print('b_multi_masks: ', b_multi_masks[0].shape)
            print('b_mask_instances: ', b_mask_instances[0].shape)
            print('b_mask_proposals: ', b_mask_proposals.size())

    def test_empty_mask(self):
        print('test mask nms_func for empty rcnn proposals')
        roi_align = RoiAlign(self.cfg, self.cfg.rcnn_crop_size)

        empty_proposal = torch.zeros((1, 7))
        crop = roi_align(self.fs, empty_proposal)

        mask_head = MaskHead(self.cfg, 256)
        logits = mask_head(crop)

        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:

            b_multi_masks, b_mask_instances, b_mask_proposals = mask_nms(self.cfg, images, empty_proposal, logits)
            print('b_multi_masks: ', b_multi_masks[0].shape)
            print('b_mask_instances: ', b_mask_instances[0].shape)
            print('b_mask_proposals: ', b_mask_proposals.size())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestNms("test_rpn"),
        TestNms("test_rcnn"),
        TestNms("test_empty_rcnn"),
        TestNms("test_mask"),
        TestNms("test_empty_mask")
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)