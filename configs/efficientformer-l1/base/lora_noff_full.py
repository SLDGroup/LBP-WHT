gradient_filter = dict(
    enable=True,
    filter_install=[
        dict(path="backbone.network.3.5",
             type='efficientformer-attn-lora-noff', radius=8, extra_cfg=dict(alpha=8)),
    ]
)
