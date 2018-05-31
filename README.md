# Mask RCNN PyTorch
PyTorch 0.4 implementation of Mask-RCNN.

Features:

1. Device-agnostic code. Run with both gpu/cpu without modifying the code, gpu is not necessary for both train and test. Thanks to pytorch 0.4!
3. Full-documented code, with jupyter notebook guidance, easy-to-use configuration
4. Clear code structure with full unit test, with minimal pain to extend

## Requirements

Ubuntu 16.04/macOS High Sierra

 - [Anaconda 3](https://anaconda.org)
 - [PyTorch 0.4.0](https://pytorch.org)
 - [python opencv 3](https://pypi.org/project/opencv-python/)
 - [Cython](http://cython.org)

Optional:
 - GPU with CUDA support

```bash
cd /tmp
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
# restart terminal
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

sudo apt-get install -y python-qt4
sudo apt-get install -y libsm6 libxext6
pip3 install opencv-python
```

## Install

```
git clone https://github.com/shawnau/mask_rcnn_pytorch.git
cd mask_rcnn_pytorch
```

### Get GPU Arch (Optinal for gpu users)
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list

Modify these 2 files according to your gpu's arch
 - [build_gpu.sh](https://github.com/shawnau/mask_rcnn_pytorch/blob/0c26d5dfaedbdf8ada0f96163a1e1f4103c2a843/build_gpu.sh#L11)
 - [net/lib/roi_align/src/Makefile](https://github.com/shawnau/mask_rcnn_pytorch/blob/0c26d5dfaedbdf8ada0f96163a1e1f4103c2a843/net/lib/roi_align/src/Makefile#L2)

### Build layers
```
./build_gpu.sh # for gpu user
./build_cpu.sh # for cpu user
```

## Use

### Test
See `demo.ipynb`

### Train
See `train.py`

## Development

Code Structure
```
net
├── layer
│   ├── backbone
│   │   └── SE_ResNeXt_FPN.py  ResNeXt50-FPN with SE-Scale
│   ├── mask
│   │   ├── mask_head.py    Mask Network
│   │   ├── mask_target.py  For training Mask head
│   │   └── mask_utils.py   Mask loss and helper functions
│   ├── nms.py              Non max suppression with bbox regression
│   ├── rcnn
│   │   ├── rcnn_head.py    RCNN Network
│   │   ├── rcnn_target.py  For training RCNN head
│   │   └── rcnn_utils.py   RCNN loss
│   ├── roi_align.py        ROI Align
│   └── rpn
│       ├── rpn_head.py     RPN Network
│       ├── rpn_target.py   For training RPN head
│       └── rpn_utils.py    RPN loss, anchor boxes generating
├── lib
│   ├── box_overlap         Canculationg box overlap, ported from fast-rcnn
│   ├── cython_nms          nms used for cpu, ported from faster-rcnn
│   ├── gpu_nms             nms used for gpu, ported from faster-rcnn
│   └── roi_align           RoiAlign ported from tensorflow crop and resize layer
├── mask_rcnn.py
└── utils
    ├── box_utils.py        box transformations
    ├── draw.py             draw boxes and masks
    ├── file.py             io
    └── func_utils.py       loss functions
```