_base_ = [
    "full_swinv2-small_cifar100.py",
    "base/lora_full.py"
]

freeze_layers = ['~backbone.lora']
