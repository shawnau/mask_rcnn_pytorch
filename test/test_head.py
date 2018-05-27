import numpy as np
import torch
import unittest

from net.layer.backbone.SE_ResNeXt_FPN import SEResNeXtFPN
from net.layer.rpn.rpn_head import RpnMultiHead
from net.layer.rpn.rpn_utils import rpn_make_anchor_boxes
from net.layer.roi_align.crop import CropRoi
from net.layer.rcnn.rcnn_head import RcnnHead
from net.layer.mask.mask_head import MaskHead


class Configuration:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = 'test_data/'
        # data parameters
        self.num_classes = 2
        # rpn parameters
        self.rpn_base_sizes = [8, 16, 32, 64]
        self.rpn_scales = [2, 4, 8, 16]
        aspect = lambda s, x: (s * 1 / x ** 0.5, s * x ** 0.5)
        self.rpn_base_apsect_ratios = [
            [(1, 1), aspect(2 ** 0.5, 2), aspect(2 ** 0.5, 0.5), ],
            [(1, 1), aspect(2 ** 0.5, 2), aspect(2 ** 0.5, 0.5), ],
            [(1, 1), aspect(2 ** 0.5, 2), aspect(2 ** 0.5, 0.5), ],
            [(1, 1), aspect(2 ** 0.5, 2), aspect(2 ** 0.5, 0.5), ],
        ]
        # rcnn parameters
        self.rcnn_crop_size = 14


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
        print("logits_flat: ", logits_flat.size())
        print("deltas_flat: ", deltas_flat.size())

    def test_rpn_anchors(self):
        rpn_anchor_boxes = rpn_make_anchor_boxes(self.fs, self.cfg)
        print("rpn_anchor_boxes: ", len(rpn_anchor_boxes))


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
        self.net = CropRoi(self.cfg, self.cfg.rcnn_crop_size)

    def test_forward(self):
        # [i, x0, y0, x1, y1, score, label]
        proposals = torch.FloatTensor([[0, 1, 1, 5, 5, 0.6, 1],
                                       [1, 2, 2, 6, 6, 0.2, 0],
                                       [2, 3, 3, 7, 7, 0.5, 1],])
        crops = self.net(self.fs, proposals)
        print("cropped features: ", crops.size())

    def test_empty(self):
        empty = torch.zeros((1, 7))
        crops = self.net(self.fs, empty)
        print("empty features: ", crops.size())


class TestRCNNHead(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        self.crop = torch.randn(5, 256, 14, 14)
        print('=' * 10 + 'Test RCNN Head' + '=' * 10)
        self.net = RcnnHead(self.cfg, 256)

    def test_forward(self):

        logits, deltas = self.net(self.crop)

        print("logits: ", logits.size())
        print("deltas: ", deltas.size())


class TestMaskHead(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        self.crop = torch.randn(5, 256, 14, 14)
        print('=' * 10 + 'Test Mask Head' + '=' * 10)
        self.net = MaskHead(self.cfg, 256)

    def test_forward(self):
        logits = self.net(self.crop)
        print("logits: ", logits.size())


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
