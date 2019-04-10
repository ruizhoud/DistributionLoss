# Distribution_Loss
Source code for [paper](https://arxiv.org/abs/1904.02823) "Regularizing Activation Distribution for Training Binarized Deep Networks"

Code modified from the [code](https://github.com/itayhubara/BinaryNet.pytorch) for the original BNN paper.

## Train BNN-DL for Alexnet ImageNet
CUDA_VISIBLE_DEVICES=0 python custom_main_binary_imagenet.py --seed 1 --model alexnet_binary_vs_xnor --batch-size 256 --batch_size_test 100 --infl_ratio 1 --distrloss 2. --distr_epoch 60 --epochs 65 --gpus 0 

You can download checkpoint from this [link](https://www.dropbox.com/s/pvsujjbwdj92aj8/checkpoint.pth.tar?dl=0). The top-1 accuracy is 47.9%, and top-5 accuracy is 71.9%.
