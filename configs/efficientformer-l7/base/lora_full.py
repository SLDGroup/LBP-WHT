gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='efficientformer-4d-lora',
                    radius=8, extra_cfg=dict(alpha=8)),
    filter_install=[
        dict(path="backbone.network.0.0"),
        dict(path="backbone.network.0.1"),
        dict(path="backbone.network.0.2"),
        dict(path="backbone.network.0.3"),
        dict(path="backbone.network.0.4"),
        dict(path="backbone.network.0.5"),

        dict(path="backbone.network.1.1"),
        dict(path="backbone.network.1.2"),
        dict(path="backbone.network.1.3"),
        dict(path="backbone.network.1.4"),
        dict(path="backbone.network.1.5"),
        dict(path="backbone.network.1.6"),
    ] +
    [
        dict(path=f"backbone.network.2.{i}") for i in range(1, 19)
    ] +
    [
        dict(path="backbone.network.3.2", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.3", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.4", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.5", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.6", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.7", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.8", type='efficientformer-attn-lora'),
        dict(path="backbone.network.3.9", type='efficientformer-attn-lora'),
    ]
)
