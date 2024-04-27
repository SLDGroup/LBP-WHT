_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/cars.py",
    "../schedules/cars.py"
]

model = dict(head=dict(num_classes=196))
