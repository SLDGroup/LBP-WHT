_base_ = [
    "full_swinv2-small_cifar100.py",
    "base/lora_full.py"
]

freeze_layers = ['backbone', '~backbone.stages.3', '~backbone.stages.3.lora']
