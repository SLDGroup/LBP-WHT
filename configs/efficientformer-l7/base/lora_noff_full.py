gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='efficientformer-attn-lora-noff',
                    radius=8, extra_cfg=dict(alpha=8)),
    filter_install=[
        dict(path="backbone.network.3.2"),
        dict(path="backbone.network.3.3"),
        dict(path="backbone.network.3.4"),
        dict(path="backbone.network.3.5"),
        dict(path="backbone.network.3.6"),
        dict(path="backbone.network.3.7"),
        dict(path="backbone.network.3.8"),
        dict(path="backbone.network.3.9"),
    ]
)
