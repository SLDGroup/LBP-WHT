gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='swinv2blk_attn', radius=8,
                    extra_cfg=dict(lp_tri_order=3)),
    filter_install=[
        dict(path="backbone.stages.3.blocks.1"),
        dict(path="backbone.stages.3.blocks.0"),
        dict(path="backbone.stages.3.downsample.reduction", type='linear_wht'),
    ] + [dict(path=f"backbone.stages.2.blocks.{i}") for i in range(18)] + [
        dict(path='backbone.stages.2.downsample.reduction', type='linear_wht'),
        dict(path="backbone.stages.1.blocks.1"),
        dict(path="backbone.stages.1.blocks.0"),
        dict(path='backbone.stages.1.downsample.reduction', type='linear_wht'),
        dict(path="backbone.stages.0.blocks.1"),
        dict(path="backbone.stages.0.blocks.0"),
    ]
)
