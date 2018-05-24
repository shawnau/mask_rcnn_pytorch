import torch


class Configuration(object):
    def __init__(self):
        super(Configuration, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.version = 'SE-FPN-ResNeXt50'
        self.data_dir = 'test/data/'
        # net
        # include background class
        self.num_classes = 2

        # multi-rpn  --------------------------------------------------------
        # base size of the anchor box on input image (2*a, diameter?)
        self.rpn_base_sizes = [8, 16, 32, 64]
        # 4 dirrerent zoom scales from each feature map to input.
        # used to get stride of anchor boxes
        # e.g. 2 for p1, 4 for p2, 8 for p3, 16 for p4.
        # the smaller the feature map is, the bigger the anchor box will be
        self.rpn_scales = [2, 4, 8, 16]

        aspect = lambda s,x: (s*1/x**0.5,s*x**0.5)
        self.rpn_base_apsect_ratios = [
           [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
           [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
           [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
           [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        ]

        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.5

        self.rpn_train_scale_balance = False

        self.rpn_train_nms_pre_score_threshold = 0.50
        self.rpn_train_nms_overlap_threshold   = 0.85  # higher for more proposals for mask training
        self.rpn_train_nms_min_size = 5

        self.rpn_test_nms_pre_score_threshold = 0.60
        self.rpn_test_nms_overlap_threshold   = 0.75
        self.rpn_test_nms_min_size = 5

        # rcnn ------------------------------------------------------------------
        self.rcnn_crop_size         = 14
        self.rcnn_train_batch_size  = 32  # per image
        self.rcnn_train_fg_fraction = 0.5
        self.rcnn_train_fg_thresh_low  = 0.5
        self.rcnn_train_bg_thresh_high = 0.5
        self.rcnn_train_bg_thresh_low  = 0.0

        self.rcnn_train_nms_pre_score_threshold = 0.05
        self.rcnn_train_nms_overlap_threshold   = 0.85  # high for more proposals for mask
        self.rcnn_train_nms_min_size = 8

        self.rcnn_test_nms_pre_score_threshold = 0.50
        self.rcnn_test_nms_overlap_threshold   = 0.85
        self.rcnn_test_nms_min_size = 8

        # mask ------------------------------------------------------------------
        self.mask_crop_size            = 14  # input of mask head
        self.mask_train_batch_size     = 32  # per image
        self.mask_size                 = 28  # out put of mask head
        self.mask_train_min_size       = 8
        self.mask_train_fg_thresh_low  = self.rpn_train_fg_thresh_low

        self.mask_test_nms_pre_score_threshold = 0.1
        self.mask_test_nms_overlap_threshold = 0.2
        self.mask_test_mask_threshold  = 0.5
        self.mask_test_mask_min_area = 8

        # optim -----------------------------------------------------------------
        self.lr = 0.01
        self.iter_accum = 1  # learning rate = lr/iter_accum
        self.batch_size = 10
        self.num_iters = 2000
        self.iter_smooth = 20  # calculate smoothed loss over each 20 iter
