_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/cifar10.py",
    "../schedules/cifar10.py"
]

model = dict(head=dict(num_classes=10))
