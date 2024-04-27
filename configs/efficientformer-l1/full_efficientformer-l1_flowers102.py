_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/flowers102.py",
    "../schedules/flowers102.py"
]

model = dict(head=dict(num_classes=102))
