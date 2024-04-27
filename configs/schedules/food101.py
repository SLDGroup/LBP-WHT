runner = dict(type="EpochBasedRunner", max_epochs=50)
evaluation = dict(gpu_collect=True, save_best='accuracy_top-1')
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=1000,
    warmup_by_epoch=False)
checkpoint_config = dict(interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])
