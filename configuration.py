import os
import torch


class Configuration(object):
    def __init__(self):
        super(Configuration, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.version = 'SE-FPN-ResNeXt50'
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'coco2017')
        # net ---------------------------------------------------------------
        # number of the classes, including background class 0
        self.num_classes = 80 + 1

        # multi-rpn  --------------------------------------------------------
        # base size of the anchor box on input image
        self.rpn_base_sizes = [32, 64, 128, 256, 512]
        self.rpn_scales = [4, 8, 16, 32, 64]
        # anchor aspects. please referring to the doc
        aspect = lambda s, r: (s * 1 / r ** 0.5, s * r ** 0.5)
        self.rpn_base_aspect_ratios = [
           [(1, 1), aspect(2**0.5, 2), aspect(2**0.5, 0.5), ],
           [(1, 1), aspect(2**0.5, 2), aspect(2**0.5, 0.5), ],
           [(1, 1), aspect(2**0.5, 2), aspect(2**0.5, 0.5), ],
           [(1, 1), aspect(2**0.5, 2), aspect(2**0.5, 0.5), ],
           [(1, 1), aspect(2**0.5, 2), aspect(2**0.5, 0.5), ],
        ]
        # background: 0.0 < overlap < 0.5
        # foreground: 0.5 < overlap < 1.0
        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.5

        # select anchor boxes with
        # score > 0.5, overlap > 0.85 , size > 5 pixel^2 as nms output for training
        self.rpn_train_nms_pre_score_threshold = 0.70
        self.rpn_train_nms_overlap_threshold   = 0.85
        self.rpn_train_nms_min_size = 5
        # same as training
        self.rpn_test_nms_pre_score_threshold = 0.70
        self.rpn_test_nms_overlap_threshold   = 0.85
        self.rpn_test_nms_min_size = 5

        # rcnn ------------------------------------------------------------------
        self.rcnn_crop_size         = 14  # roi align pooling size
        self.rcnn_train_batch_size  = 32  # rcnn proposals for training per image
        self.rcnn_train_fg_fraction = 0.5     # foreground fraction for training
        self.rcnn_train_fg_thresh_low  = 0.5  # 0.5 < overlap < 1.0 as foreground
        self.rcnn_train_bg_thresh_high = 0.5  # thresh_low < overlap < thresh_high as background
        self.rcnn_train_bg_thresh_low  = 0.0  # same as above
        # same as rpn
        self.rcnn_train_nms_pre_score_threshold = 0.05
        self.rcnn_train_nms_overlap_threshold   = 0.85
        self.rcnn_train_nms_min_size = 8

        self.rcnn_test_nms_pre_score_threshold = 0.70
        self.rcnn_test_nms_overlap_threshold   = 0.85
        self.rcnn_test_nms_min_size = 8

        # mask ------------------------------------------------------------------
        self.mask_crop_size            = 14  # input of mask head
        self.mask_train_batch_size     = 32  # mask proposals for training per image
        self.mask_size                 = 28  # output size of mask head
        self.mask_train_min_size       = 8
        self.mask_train_fg_thresh_low  = self.rpn_train_fg_thresh_low
        # same as rpn
        self.mask_test_nms_pre_score_threshold = 0.3
        self.mask_test_nms_overlap_threshold = 0.2
        self.mask_test_mask_threshold  = 0.5  # mask binary threshold
        self.mask_test_mask_min_area = 8

        # optim -----------------------------------------------------------------
        self.lr = 0.001
        self.iter_accum = 1  # learning rate = lr/iter_accum
        self.batch_size = 1
        self.num_iters = int(100000 / self.batch_size * 50)
        self.iter_smooth = 1  # calculate smoothed loss over each 20 iter
        self.iter_valid = 100
        self.iter_save = self.num_iters / 10

        # checkpoint
        self.checkpoint = None
