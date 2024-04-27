#!/bin/bash

for ds in cifar10 cifar100
do
    for ord in 2 4 8
    do
        echo Train EfficientFormer-L1 on $ds with LP_L1-$ord
        python train.py configs/efficientformer-l1/filt_lptri_layer4_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth --cfg-options gradient_filter.common_cfg.extra_cfg.lp_tri_order=$ord --log-postfix ord$ord
    done
    echo Train EfficientFormer-L1 on $ds with Lora-all
    python train.py configs/efficientformer-l1/lora_layer4_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth --log-postfix ord$ord
    echo Train EfficientFormer-L1 on $ds with Lora
    python train.py configs/efficientformer-l1/lora_noff_layer4_efficientformer-l1_$ds.py --load-from pretrained_ckpts/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth --log-postfix ord$ord
done
