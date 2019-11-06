#!/bin/bash

. path.sh

ckp=exp/10fsmnh1024p256l20r20s22/checkpoint-17
hybrid=true
train=data/feature/train
valid=data/feature/valid
transform=data/final.feature_transform
cmvn='--norm-means=true --norm-vars=false'
splice='0 0'
delay=5
n_gpu=1
show=false

. utils/parse_options.sh

echo 'start training'

python -u rl/train.py \
    --ckp $ckp \
    --hybrid $hybrid \
    --train-data-dir $train \
    --cv-data-dir $valid \
    --feature-transform $transform \
    --cmvn-opts "$cmvn" \
    --splice $splice \
    --targets-delay $delay \
    --n-gpu $n_gpu \
    --show $show \
