import unittest
from torch.utils.data import DataLoader
from loader.dsb2018.train_utils import *

from configuration import Configuration
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead

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
        self.cfg.data_dir = 'test_data/'
        # loader
        train_dataset = ScienceDataset(self.cfg, mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=make_collate)


    def test_rpn(self):
        p2 = torch.randn(5, 256, 128, 128)
        p3 = torch.randn(5, 256, 64, 64)
        p4 = torch.randn(5, 256, 32, 32)
        p5 = torch.randn(5, 256, 16, 16)
        fs = [p2, p3, p4, p5]

        net = RpnMultiHead(self.cfg, 256)
        logits_flat, deltas_flat = net(fs)

        print('=' * 10, 'Test RPN nms', '=' * 10)
        for images, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            images: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """

            anchor_boxes = rpn_make_anchor_boxes(fs, self.cfg)
            rpn_proposals = rpn_nms(self.cfg, 'train', images, anchor_boxes, logits_flat, deltas_flat)
            print('rpn_proposals: ', rpn_proposals.size())

    def test_rcnn(self):
        print('=' * 10, 'Test RCNN nms', '=' * 10)
        rpn_proposals = np.array(
            [[0., 52.,  161., 70.,  176., 0.6, 1.],
             [0., 148., 190., 176., 215., 0.7, 1.],
             [0., 130., 30.,  150., 60.,  0.5, 1.],
             [0., 250., 170., 240., 200., 0.7, 1.],
             [0., 40.,  90.,  63.,  100., 0.8, 1.],
             [0., 170., 120., 189., 108., 0.9, 1.]]
        ).astype(np.float32)  # overlap must use float32
        rpn_proposals = torch.from_numpy(rpn_proposals).to(self.cfg.device)

        crop = torch.randn(6, 256, 14, 14)  # 6 rpn_proposals got from rpn nms
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
            rcnn_proposals = rcnn_nms(self.cfg, 'train', images, rpn_proposals, logits, deltas)
            print('rcnn_proposals: ', rcnn_proposals.size())

    def test_mask(self):
        print('=' * 10, 'Test Mask nms', '=' * 10)
        rcnn_proposals = np.array(
            [[0., 52.,  161., 70.,  176., 0.6, 1.],
             [0., 148., 190., 176., 215., 0.7, 1.],
             [0., 130., 30.,  150., 60.,  0.5, 1.],
             [0., 250., 170., 255., 200., 0.7, 1.],
             [0., 40.,  90.,  63.,  100., 0.8, 1.],
             [0., 170., 120., 189., 128., 0.9, 1.]]
        ).astype(np.float32)  # overlap must use float32
        rcnn_proposals = torch.from_numpy(rcnn_proposals).to(self.cfg.device)

        crop = torch.randn(6, 256, 14, 14)  # 6 rcnn_proposals got from rpn nms
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
            b_multi_masks, b_mask_instances, b_mask_proposals = mask_nms(self.cfg, images, rcnn_proposals, logits)
            print('b_multi_masks: ', b_multi_masks[0].shape)
            print('b_mask_instances: ', b_mask_instances[0].shape)
            print('b_mask_proposals: ', b_mask_proposals.size())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestNms("test_rpn"),
        TestNms("test_rcnn"),
        TestNms("test_mask")
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)