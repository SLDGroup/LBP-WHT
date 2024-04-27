dataset_type = 'Food101'
img_norm_cfg = dict(
    mean=[87.6768, 113.1902, 139.0707],
    std=[70.9603, 69.6332, 68.8928],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/food-101/images',
        ann_file="data/food-101/meta/train.txt",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/food-101/images',
        ann_file="data/food-101/meta/test.txt",
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/food-101/images',
        ann_file="data/food-101/meta/test.txt",
        pipeline=test_pipeline,
        test_mode=True))
