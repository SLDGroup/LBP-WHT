_base_ = [
    "full_efficientformer-l1_cifar10.py",
    "base/lora_full.py"
]

freeze_layers = ['~backbone.lora']
