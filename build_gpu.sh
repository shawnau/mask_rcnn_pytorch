#!/usr/bin/env bash

cd net/lib/box_overlap
python setup.py build_ext --inplace

cd ../gpu_nms
python setup.py build_ext --inplace

cd ../roi_align
cd src
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ..
python build.py
