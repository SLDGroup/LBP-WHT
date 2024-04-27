gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='swinv2blk_attn_lora', radius=8,
                    extra_cfg=dict(alpha=8)),
    filter_install=[
        dict(path="backbone.stages.3.blocks.1"),
        dict(path="backbone.stages.3.blocks.0"),
        dict(path="backbone.stages.3.downsample.reduction", type='linear_lora'),
    ] + [dict(path=f"backbone.stages.2.blocks.{i}") for i in range(18)] + [
        dict(path='backbone.stages.2.downsample.reduction', type='linear_lora'),
        dict(path="backbone.stages.1.blocks.1"),
        dict(path="backbone.stages.1.blocks.0"),
        dict(path='backbone.stages.1.downsample.reduction', type='linear_lora'),
        dict(path="backbone.stages.0.blocks.1"),
        dict(path="backbone.stages.0.blocks.0"),
    ]
)
