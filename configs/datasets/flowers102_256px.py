dataset_type = 'Flowers102'
img_norm_cfg = dict(
    mean=[73.3100, 96.2098, 110.9518],
    std=[68.7424, 62.4863, 75.5192],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(type='RandomResizedCrop', size=256),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/flowers-102/jpg',
        ann_file="data/flowers-102/train.txt",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/flowers-102/jpg',
        ann_file="data/flowers-102/test.txt",
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/flowers-102/jpg',
        ann_file="data/flowers-102/test.txt",
        pipeline=test_pipeline,
        test_mode=True))
