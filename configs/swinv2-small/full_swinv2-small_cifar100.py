_base_ = [
    "base/swinv2-small.py",
    "../datasets/cifar100_256px.py",
    "../schedules/cifar100.py"
]

model = dict(head=dict(num_classes=100))
