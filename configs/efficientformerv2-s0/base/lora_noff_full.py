gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='efficientformerv2-attn-lora-noff',
                    radius=8, extra_cfg=dict(alpha=8)),
    filter_install=[dict(path=f"backbone.network.6.{i}") for i in range(2, 4)] +
    [
        dict(path="backbone.network.5.attn.q.proj.0",
             type='pointwise_conv_lora'),
        dict(path="backbone.network.5.attn.k.0",
             type='pointwise_conv_lora'),
        dict(path="backbone.network.5.attn.v.0",
             type='pointwise_conv_lora'),
        dict(path="backbone.network.5.attn.proj.1",
             type='pointwise_conv_lora'),
    ] +
    [dict(path=f"backbone.network.4.{i}") for i in range(4, 6)]
)
