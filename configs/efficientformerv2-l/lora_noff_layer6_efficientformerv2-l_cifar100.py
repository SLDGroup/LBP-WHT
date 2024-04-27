_base_ = [
    "full_efficientformerv2-l_cifar100.py",
    "base/lora_noff_full.py"
]

freeze_layers = ['backbone', '~backbone.network.6.lora']
