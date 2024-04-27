_base_ = "../../../mmclassification/configs/_base_/models/swin_transformer_v2/small_256.py"

model = dict(
    backbone=dict(norm_eval=True),
    head=dict(num_classes=100),
    train_cfg=None
)
