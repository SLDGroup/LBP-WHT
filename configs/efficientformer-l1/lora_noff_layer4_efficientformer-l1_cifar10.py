_base_ = [
    "full_efficientformer-l1_cifar10.py",
    "base/lora_noff_full.py"
]

freeze_layers = ['backbone', '~backbone.network.3.lora']
