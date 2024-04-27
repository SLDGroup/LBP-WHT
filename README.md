# LBP-WHT: Efficient Low-rank Backpropagation for Vision Transformer Adaptation

> Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu

<details><summary>Abstract</summary>
The increasing scale of vision transformers (ViT) has made the efficient fine-tuning of these large models for specific needs a significant challenge in various applications. This issue originates from the computationally demanding matrix multiplications required during the backpropagation process through linear layers in ViT. In this paper, we tackle this problem by proposing a new Low-rank Back-Propagation via Walsh-Hadamard Transformation (LBP-WHT) method. Intuitively, LBP-WHT projects the gradient into a low-rank space and carries out backpropagation. This approach substantially reduces the computation needed for adapting ViT, as matrix multiplication in the low-rank space is far less resource-intensive. We conduct extensive experiments with different models (ViT, hybrid convolution-ViT model) on multiple datasets to demonstrate the effectiveness of our method. For instance, when adapting an EfficientFormer-L1 model on CIFAR100, our LBP-WHT achieves 10.4% higher accuracy than the state-of-the-art baseline, while requiring 9 MFLOPs less computation. As the first work to accelerate ViT adaptation with low-rank backpropagation, our LBP-WHT method is complementary to many prior efforts and can be combined with them for better performance.

</details>

## Setup
- Python 3.8
- PyTorch 1.13.1
- Install MMClassification:
    ```
    pip install openmim
    mim install mmcv-full==1.7.0
    cd mmclassification
    pip install -e .
    pip install yapf==0.32.0
    pip install scipy future tensorboard
    ```
- Prepare datasets:
    - Create a `data` folder under the root of the repo
    - Cifar10/100 will be download and preprocessed automatically.
    - For Flowers102, Food101, Pets, Stanford-Cars, you need to manually download and place them under the `data` folder
        - The Flowers102 dataset need to be preprocessed with `tools/convert_flowers_102.py`

## Get pretrained model

Config name:
- EfficientFormer L1 and L7:
    - Run: `mim download mmcls --config efficientformer-l[1, 7]_3rdparty_8xb128_in1k --dest pretrained_ckpts`
    - Need to convert the downloaded checkpoint with `tools/correct_efficientformer_ckpt.py`
- EfficientFormerV2 S0 and L: Need to download manually from [EfficientFormerV2 repo](https://github.com/snap-research/EfficientFormer)
    - S0: [link](https://drive.google.com/file/d/1PXb7b9pv9ZB4cfkRkYEdwgWuVwvEiazq/view?usp=share_link)
    - L: [link](https://drive.google.com/file/d/1sRXNBHl_ewHBMwwYZVsOJo1k6JpcNJn-/view?usp=share_link)
- SwinV2
    - Run: `mim download mmcls --config swinv2-small-w8_3rdparty_in1k-256px --dest pretrained_ckpts`

## Train
- Run `python train.py <config> <other args>` with configs under `configs/<model name>/`
- See examples in `train_example.sh`
- Commonly used args:
    - `--load-from`: path to pretrained checkpoints
    - `--cfg-options`: overwrite config, for example
        - `--cfg-options data.samples_per_gpu=128`: set batch size per gpu to 128
        - `--cfg-options data.samples_per_gpu=128 optimizer.lr=1e-4`: set batch size per gpu to 128 and learning rate to 1e-4
    - `--log-postfix`: postfix to the log dir

- Training logs will be placed under `runs-<git branch name>/<config name>_<log-postfix>/` unless specified in args

## Credit

This repo is developed based on [MMClassification](https://github.com/open-mmlab/mmpretrain) (now MMPretrain).

## License

See [LICENSE](LICENSE)
