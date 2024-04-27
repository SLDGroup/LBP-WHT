_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/pets.py",
    "../schedules/pets.py"
]

model = dict(head=dict(num_classes=37))
