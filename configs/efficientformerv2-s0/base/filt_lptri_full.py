gradient_filter = dict(
    enable=True,
    common_cfg=dict(type='efficientformerv2-conv', radius=8,
                    extra_cfg=dict(lp_tri_order=3)),
    filter_install=[dict(path=f"backbone.network.6.{i}", type='efficientformerv2-attn') for i in range(2, 4)] +
    [dict(path=f"backbone.network.6.{i}") for i in range(2)] +
    [
        dict(path="backbone.network.5.attn.q.proj.0", type='pointwise_conv_wht'),
        dict(path="backbone.network.5.attn.k.0", type='pointwise_conv_wht'),
        dict(path="backbone.network.5.attn.v.0", type='pointwise_conv_wht'),
        dict(path="backbone.network.5.attn.proj.1", type='pointwise_conv_wht'),
    ] +
    [dict(path=f"backbone.network.4.{i}", type='efficientformerv2-attn') for i in range(4, 6)] +
    [dict(path=f"backbone.network.4.{i}") for i in range(4)] +
    [dict(path=f"backbone.network.2.{i}") for i in range(2)] +
    [dict(path=f"backbone.network.0.{i}") for i in range(2)]
)
