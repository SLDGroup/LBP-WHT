_base_ = [
    'full_efficientformerv2-s0_cifar100.py',
    'base/filt_lptri_full.py'
]

gradient_filter = dict(common_cfg=dict(extra_cfg=dict(lp_tri_order=8)))
