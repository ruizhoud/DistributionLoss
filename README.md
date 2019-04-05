# Distribution_Loss
Source code for paper "Regularizing Activation Distribution for Training Binarized Deep Networks"

Code modified from the [code](https://github.com/itayhubara/BinaryNet.pytorch) for the original BNN paper.

# Train BNN-DL for Alexnet ImageNet
CUDA_VISIBLE_DEVICES=0 python custom_main_binary_imagenet.py --seed 1 --model alexnet_binary_vs_xnor --batch-size 256 --batch_size_test 100 --infl_ratio 1 --distrloss 2. --distr_epoch 60 --epochs 67 --gpus 0 