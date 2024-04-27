_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/cifar100.py",
    "../schedules/cifar100.py"
]

model = dict(head=dict(num_classes=100))
