_base_ = [
    "full_swinv2-small_cifar100.py",
    "base/filt_lptri_full.py"
]

gradient_filter = dict(common_cfg=dict(extra_cfg=dict(lp_tri_order=8)))
