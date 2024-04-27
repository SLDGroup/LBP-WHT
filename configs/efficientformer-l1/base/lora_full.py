gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='efficientformer-4d-lora',
                    radius=8, extra_cfg=dict(alpha=8)),
    filter_install=[
        dict(path="backbone.network.0.0"),
        dict(path="backbone.network.0.1"),
        dict(path="backbone.network.0.2"),
        dict(path="backbone.network.1.1"),
        dict(path="backbone.network.1.2"),
        dict(path="backbone.network.2.1"),
        dict(path="backbone.network.2.2"),
        dict(path="backbone.network.2.3"),
        dict(path="backbone.network.2.4"),
        dict(path="backbone.network.2.5"),
        dict(path="backbone.network.2.6"),
        dict(path="backbone.network.3.1"),
        dict(path="backbone.network.3.2"),
        dict(path="backbone.network.3.3"),
        dict(path="backbone.network.3.5", type='efficientformer-attn-lora'),
    ]
)
