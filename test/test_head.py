import torch
import unittest

from configuration import Configuration

from net.layer.backbone.SE_ResNeXt_FPN import SEResNeXtFPN
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes
from net.layer.roi_align import RoiAlign
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead


class TestBackbone(unittest.TestCase):
    def check_output_size(self, net):
        """
        register forward hook to print main layer's input/output size for testing
        :param net:
        :type nn.Module
        :return:
        """

        def hook(self, input, output):
            # input is a tuple of packed inputs
            # output is a Tensor. output.data is the Tensor we are interested
            print('=' * 10 + self.__class__.__name__ + '=' * 10)
            print('input size:', input[0].size())
            print('output size:', output.size())

        names = [name for name, _ in net.named_modules()]
        first_level_modules = list(set([x.split('.')[0] for x in names if x]))
        print(first_level_modules)
        for name, module in net.named_modules():
            if name in first_level_modules:
                module.register_forward_hook(hook)

    def setUp(self):
        # batch_size=5, channel=3, height=256, width=256
        self.input_tensor = torch.randn(5, 3, 256, 256)
        print('=' * 10 + 'Test Backbone' + '=' * 10)
        self.net = SEResNeXtFPN([3, 4, 6, 3])

    def test_forward(self):
        self.check_output_size(self.net)
        out = self.net(self.input_tensor)
        print('='*10, 'Final output: ', '='*10)
        for tensor in out:
            print(tensor.size())


class TestRPNHead(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # input features
        p2 = torch.randn(5, 256, 128, 128)
        p3 = torch.randn(5, 256, 64, 64)
        p4 = torch.randn(5, 256, 32, 32)
        p5 = torch.randn(5, 256, 16, 16)
        self.fs = [p2, p3, p4, p5]
        print('=' * 10 + 'Test RPN Head' + '=' * 10)
        self.net = RpnMultiHead(self.cfg, 256)

    def test_forward(self):
        logits_flat, deltas_flat = self.net(self.fs)

        batch_size, num_achor, num_classes = logits_flat.size()
        print("logits_flat: ", logits_flat.size())
        self.assertEqual(batch_size, 5)
        self.assertEqual(num_achor, (16 * 16 + 32 * 32 + 64 * 64 + 128 * 128) * 3)
        self.assertEqual(num_classes, 2)

        batch_size, num_achor, num_classes, num_delta = deltas_flat.size()
        print("deltas_flat: ", deltas_flat.size())
        self.assertEqual(batch_size, 5)
        self.assertEqual(num_achor, (16 * 16 + 32 * 32 + 64 * 64 + 128 * 128) * 3)
        self.assertEqual(num_classes, 2)
        self.assertEqual(num_delta, 4)

    def test_rpn_anchors(self):
        rpn_anchor_boxes = rpn_make_anchor_boxes(self.fs, self.cfg)
        print("rpn_anchor_boxes: ", len(rpn_anchor_boxes))
        self.assertEqual(len(rpn_anchor_boxes), (16 * 16 + 32 * 32 + 64 * 64 + 128 * 128) * 3)


class TestROIAlign(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # input features
        p2 = torch.randn(5, 256, 128, 128)
        p3 = torch.randn(5, 256, 64, 64)
        p4 = torch.randn(5, 256, 32, 32)
        p5 = torch.randn(5, 256, 16, 16)
        self.fs = [p2, p3, p4, p5]
        print('=' * 10 + 'Test ROI Align' + '=' * 10)
        self.net = RoiAlign(self.cfg, self.cfg.rcnn_crop_size)

    def test_forward(self):
        # [i, x0, y0, x1, y1, score, label]
        proposals = torch.FloatTensor([[0, 1, 1, 5, 5, 0.6, 1],
                                       [1, 2, 2, 6, 6, 0.2, 0],
                                       [2, 3, 3, 7, 7, 0.5, 1],])
        crops = self.net(self.fs, proposals)
        batch_size, in_channel, crop_width, crop_height = crops.size()
        print("cropped features: ", crops.size())
        self.assertEqual(batch_size, len(proposals))
        self.assertEqual(in_channel, 256)
        self.assertEqual(crop_width, self.cfg.rcnn_crop_size)
        self.assertEqual(crop_height, self.cfg.rcnn_crop_size)

    def test_empty(self):
        empty = torch.zeros((1, 7))
        crops = self.net(self.fs, empty)
        print("Cropped zero proposals: ", crops.size())


class TestRCNNHead(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        self.crop = torch.randn(5, 256, 14, 14)
        print('=' * 10 + 'Test RCNN Head' + '=' * 10)
        self.net = RcnnHead(self.cfg, 256)

    def test_forward(self):

        logits, deltas = self.net(self.crop)
        batch_size, num_classes = logits.size()
        print("logits: ", logits.size())
        self.assertEqual(batch_size, 5)
        self.assertEqual(num_classes, self.cfg.num_classes)

        batch_size, num_deltas = deltas.size()
        print("deltas: ", deltas.size())
        self.assertEqual(batch_size, 5)
        self.assertEqual(num_deltas, self.cfg.num_classes * 4)


class TestMaskHead(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        self.crop = torch.randn(5, 256, 14, 14)
        print('=' * 10 + 'Test Mask Head' + '=' * 10)
        self.net = MaskHead(self.cfg, 256)

    def test_forward(self):
        logits = self.net(self.crop)
        batch_size, num_classes, mask_width, mask_height = logits.size()
        print("logits: ", logits.size())
        self.assertEqual(batch_size, 5)
        self.assertEqual(num_classes, self.cfg.num_classes)
        self.assertEqual(mask_width, self.cfg.mask_size)
        self.assertEqual(mask_height, self.cfg.mask_size)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestBackbone("test_forward"),
        TestRPNHead("test_forward"),
        TestRPNHead("test_rpn_anchors"),
        TestROIAlign("test_forward"),
        TestROIAlign("test_empty"),
        TestRCNNHead("test_forward"),
        TestMaskHead("test_forward")
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)
