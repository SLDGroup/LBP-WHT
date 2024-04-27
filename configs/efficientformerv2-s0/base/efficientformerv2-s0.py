model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormerV2',
        arch='S0',
        drop_path_rate=0,
        resolution=224,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['BatchNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ],
        norm_eval=True,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='EfficientFormerClsHead',
        in_channels=176,
        num_classes=100,
        distillation=False,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1),
    )
)
