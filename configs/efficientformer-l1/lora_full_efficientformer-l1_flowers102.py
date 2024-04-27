_base_ = [
    "full_efficientformer-l1_flowers102.py",
    "base/lora_full.py"
]

freeze_layers = ['~backbone.lora']
