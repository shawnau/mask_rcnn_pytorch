#!/usr/bin/env bash

cd net/lib/box_overlap
python setup.py build_ext --inplace

cd ../nms
python setup.py build_ext --inplace

cd ../roi_align
python build.py