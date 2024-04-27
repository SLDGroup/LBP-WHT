_base_ = [
    'full_efficientformerv2-s0_cifar100.py',
    'base/lora_full.py'
]

freeze_layers = ['~backbone.lora']
