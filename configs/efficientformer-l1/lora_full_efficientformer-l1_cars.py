_base_ = [
    "full_efficientformer-l1_cars.py",
    "base/lora_full.py"
]

freeze_layers = ['~backbone.lora']
