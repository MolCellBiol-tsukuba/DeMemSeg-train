#!/bin/bash

# create venv
python -m venv .venv_DeMemSeg
source .venv_DeMemSeg/bin/activate

# install dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

pip install scikit-image tifffile tensorboard seaborn

pip install cellpose==3.1.0