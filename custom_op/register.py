import logging
from copy import deepcopy
from functools import reduce
import torch.nn as nn
from .linear_wht import wrap_linear_wht_layer
from .linear_lora import wrap_linear_lora_layer
from .pointwise_conv_wht import wrap_pointwise_conv_wht_layer
from .pointwise_conv_lora import wrap_pointwise_conv_lora_layer


DEFAULT_CFG = {
    "path": "",
    "radius": 8,
    "type": "",
    "activate": True,
    "extra_cfg": dict(),
}


def add_grad_filter(module: nn.Module, cfg):
    if cfg["type"] == "linear_wht":
        module = wrap_linear_wht_layer(
            module, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "linear_lora":
        module = wrap_linear_lora_layer(
            module, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )

    elif cfg["type"] == "efficientformer-4d":
        module.mlp.fc1 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc2 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformer-4d-lora":
        module.mlp.fc1 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc2 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformer-attn":
        module.token_mixer.qkv = wrap_linear_wht_layer(
            module.token_mixer.qkv, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj = wrap_linear_wht_layer(
            module.token_mixer.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_linear_wht_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc2 = wrap_linear_wht_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformer-attn-lora":
        module.token_mixer.qkv = wrap_linear_lora_layer(
            module.token_mixer.qkv, cfg["radius"] * 3, cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj = wrap_linear_lora_layer(
            module.token_mixer.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_linear_lora_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc2 = wrap_linear_lora_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformer-attn-lora-noff":
        module.token_mixer.qkv = wrap_linear_lora_layer(
            module.token_mixer.qkv, cfg["radius"] * 3, cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj = wrap_linear_lora_layer(
            module.token_mixer.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformerv2-attn":
        module.mlp.fc2 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj[1] = wrap_pointwise_conv_wht_layer(
            module.token_mixer.proj[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.q[0] = wrap_pointwise_conv_wht_layer(
            module.token_mixer.q[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.k[0] = wrap_pointwise_conv_wht_layer(
            module.token_mixer.k[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.v[0] = wrap_pointwise_conv_wht_layer(
            module.token_mixer.v[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.talking_head1 = wrap_pointwise_conv_wht_layer(
            module.token_mixer.talking_head1,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
        module.token_mixer.talking_head2 = wrap_pointwise_conv_wht_layer(
            module.token_mixer.talking_head2,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
    elif cfg["type"] == "efficientformerv2-attn-lora":
        module.mlp.fc2 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj[1] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.proj[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.q[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.q[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.k[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.k[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.v[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.v[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.talking_head1 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head1,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
        module.token_mixer.talking_head2 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head2,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
    elif cfg["type"] == "efficientformerv2-conv":
        module.mlp.fc2 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_pointwise_conv_wht_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "efficientformerv2-attn-lora":
        module.mlp.fc2 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.proj[1] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.proj[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.q[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.q[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.k[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.k[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.v[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.v[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.talking_head1 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head1,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
        module.token_mixer.talking_head2 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head2,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
    elif cfg["type"] == "efficientformerv2-attn-lora-noff":
        module.token_mixer.proj[1] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.proj[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.q[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.q[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.k[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.k[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.v[0] = wrap_pointwise_conv_lora_layer(
            module.token_mixer.v[0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.token_mixer.talking_head1 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head1,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
        module.token_mixer.talking_head2 = wrap_pointwise_conv_lora_layer(
            module.token_mixer.talking_head2,
            cfg["radius"],
            cfg["activate"],
            cfg["extra_cfg"],
        )
    elif cfg["type"] == "efficientformerv2-conv-lora":
        module.mlp.fc2 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc2, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.mlp.fc1 = wrap_pointwise_conv_lora_layer(
            module.mlp.fc1, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "mobilevit_conv":
        module.conv[0].conv = wrap_pointwise_conv_wht_layer(
            module.conv[0].conv, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.conv[2].conv = wrap_pointwise_conv_wht_layer(
            module.conv[2].conv, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "swinv2blk_attn":
        module.attn.w_msa.qkv = wrap_linear_wht_layer(
            module.attn.w_msa.qkv, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.attn.w_msa.proj = wrap_linear_wht_layer(
            module.attn.w_msa.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.ffn.layers[0][0] = wrap_linear_wht_layer(
            module.ffn.layers[0][0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.ffn.layers[1] = wrap_linear_wht_layer(
            module.ffn.layers[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "swinv2blk_attn_lora":
        module.attn.w_msa.qkv = wrap_linear_lora_layer(
            module.attn.w_msa.qkv, cfg["radius"] * 3, cfg["activate"], cfg["extra_cfg"]
        )
        module.attn.w_msa.proj = wrap_linear_lora_layer(
            module.attn.w_msa.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.ffn.layers[0][0] = wrap_linear_lora_layer(
            module.ffn.layers[0][0], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
        module.ffn.layers[1] = wrap_linear_lora_layer(
            module.ffn.layers[1], cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "swinv2blk_attn_lora_noff":
        module.attn.w_msa.qkv = wrap_linear_lora_layer(
            module.attn.w_msa.qkv, cfg["radius"] * 3, cfg["activate"], cfg["extra_cfg"]
        )
        module.attn.w_msa.proj = wrap_linear_lora_layer(
            module.attn.w_msa.proj, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "pointwise_conv_wht":
        module = wrap_pointwise_conv_wht_layer(
            module, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    elif cfg["type"] == "pointwise_conv_lora":
        module = wrap_pointwise_conv_lora_layer(
            module, cfg["radius"], cfg["activate"], cfg["extra_cfg"]
        )
    else:
        raise NotImplementedError
    return module


def register_filter(module, cfgs):
    filter_install_cfgs = cfgs["filter_install"]
    common_cfg = cfgs.get("common_cfg", dict())
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        ccfg = deepcopy(common_cfg)
        ccfg.update(cfg)
        assert "path" in ccfg.keys()
        for k in ccfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in ccfg.keys():
                ccfg[k] = DEFAULT_CFG[k]
        path_seq = ccfg["path"].split(".")
        target = reduce(getattr, path_seq, module)
        upd_layer = add_grad_filter(target, ccfg)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)
