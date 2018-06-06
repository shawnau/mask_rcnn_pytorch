
# Mask RCNN PyTorch
PyTorch 0.4 implementation of Mask-RCNN. This was the side-project of our work in [Kaggle Data Science Bowl 2018](https://github.com/shawnau/DataScienceBowl2018)

Features:

1. Device-agnostic code. Run with both gpu/cpu without modifying the code, gpu is not necessary for both train and test. Thanks to pytorch 0.4!
3. Full-documented code, with jupyter notebook guidance, easy-to-use configuration
4. Clear code structure with full unit test, with minimal pain to extend

## Requirements

The code is tested under following system

1. Ubuntu 16.04 (CPU/Nvidia GPU)
2. macOS High Sierra (CPU version)
3. Windows 10 (CPU version), please checkout to `win10` branch

 - [Anaconda 3](https://anaconda.org)
 - [PyTorch 0.4.0](https://pytorch.org)
 - [python opencv 3](https://pypi.org/project/opencv-python/)
 - [Cython](http://cython.org) (included in anaconda)

Optional:
 - GPU with CUDA support

```bash
# install requirements
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

### Get GPU Arch for cuda compiler (Optinal for gpu users)

| model | arch |
|--|--|
| GTX-970, GTX-980, GTX Titan X | sm_52 |
| Tesla P100 | sm_60 |
| GTX 10XX, Titan Xp, Tesla P40| sm_61 |
| Tesla V100 | sm_70 |

[Official Doc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)

Modify `-arch` keyword for these 2 files according to your gpu's arch above
 - [build_gpu.sh](https://github.com/shawnau/mask_rcnn_pytorch/blob/0c26d5dfaedbdf8ada0f96163a1e1f4103c2a843/build_gpu.sh#L11)
 - [net/lib/roi_align/src/Makefile](https://github.com/shawnau/mask_rcnn_pytorch/blob/0c26d5dfaedbdf8ada0f96163a1e1f4103c2a843/net/lib/roi_align/src/Makefile#L2)

### Build layers
```
./build_gpu.sh # for gpu user
./build_cpu.sh # for cpu user
```

## Use

### Test on Data Science Bowl 2018 image
 - Download [pretrained grayscale model](https://drive.google.com/open?id=1E61LL0fPMVeYhPAZCbt3A6PbmM8vj4Ku) into project root folder.
 - Run `demo.ipynb`

### Train
1. Train Nuclei Segmentation Model of Data Science Bowl 2018
 - Download  and unzip `stage1_train.zip` from [Download Page](https://www.kaggle.com/c/data-science-bowl-2018/data), put it into `data/`
 - Run `data/convert.py` for preprocessing
 - Run  `train.py`

2. Train MS COCO Detection 2017 Dataset
 - `git checkout coco`
 - Download coco dataset, put into `data/` with following structure:

```
coco2017
    ├── annotations
    │   └── instances_train2017.json
    └── images
        └── train2017
            └── *.jpg
```
  - Run `train.py`

## Development
Project Structure: see `doc/`

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

## To-do
 - [ ] Pretrained model on MS COCO
 - [ ] Using PIL instead of opencv
 - [ ] Pure PyTorch RoiAlign, nms and overlap

## Difference from the original paper

1. We use the idea of [UnitBox](https://arxiv.org/abs/1608.01471) for rpn anchor box regression.
2. Training code resizes all the images into `512*512`, which could be improved.
