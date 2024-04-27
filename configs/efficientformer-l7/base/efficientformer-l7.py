model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormer',
        arch='l7',
        drop_path_rate=0,
        resolution=7,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ],
        norm_eval=True,
        extra_cfg=dict(split_qkv=False)
    ),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='EfficientFormerClsHead',
        in_channels=768,
        num_classes=100,
        distillation=False,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1),
    ))
