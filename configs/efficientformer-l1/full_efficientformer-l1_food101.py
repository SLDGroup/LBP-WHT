_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/food101.py",
    "../schedules/food101.py"
]

model = dict(head=dict(num_classes=101))
