_base_ = [
    "full_efficientformer-l7_cifar100.py",
    "base/lora_full.py"
]

freeze_layers = ['backbone', '~backbone.network.3', '~backbone.network.3.lora']
