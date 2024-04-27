import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn
from math import ceil
from .wht import hadamard_func, generate_order


class PointwiseConvWHTOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, order, masks = args
        b, _, h, w = x.shape
        c_o, c_i = weight.shape[:2]
        n_mask = masks.shape[0]
        masks = masks.view(n_mask, 1, 1, 1, 1, order, order)
        y = conv2d(x, weight=weight, bias=bias)
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_pad_t, x_pad_l = ceil((p_h * order - h) / 2), ceil((p_w * order - w) / 2)
        x_pad_b, x_pad_r = p_h * order - h - x_pad_t, p_w * order - w - x_pad_l
        pad_seq = x_pad_l, x_pad_r, x_pad_t, x_pad_b
        x_pad = pad(x, pad_seq, mode="reflect")
        x_patch = (
            x_pad.view(b, c_i, p_h, order, p_w, order)
            .permute(0, 1, 2, 4, 3, 5)
            .unsqueeze(0)
        )
        x_sum = (x_patch * masks).sum(dim=(-1, -2))
        cfgs = th.tensor([bias is not None, h, w, order])
        pad_seq = th.tensor(pad_seq)
        ctx.save_for_backward(x_sum, weight, cfgs, pad_seq, masks)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight, cfgs, pad_seq, masks = ctx.saved_tensors
        has_bias, h, w, order = cfgs
        pad_seq = [int(p) for p in pad_seq]
        (grad_y,) = grad_outputs
        n_mask, b, _, p_h, p_w = x_sum.shape
        c_o, c_i = weight.shape[:2]
        order = int(order)
        grad_y_pad = pad(grad_y, pad_seq, mode="reflect")
        grad_y_patch = (
            grad_y_pad.view(b, c_o, p_h, order, p_w, order)
            .permute(0, 1, 2, 4, 3, 5)
            .unsqueeze(0)
        )
        grad_y_avg = (grad_y_patch * masks).mean(dim=(-1, -2))

        grad_x_avg = conv2d(
            grad_y_avg.view(n_mask * b, c_o, p_h, p_w), weight.permute(1, 0, 2, 3)
        )
        grad_x = grad_x_avg.view(n_mask, b, c_i, p_h, p_w, 1, 1) * masks
        grad_x_rec = (
            grad_x.sum(dim=0)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c_i, p_h * order, p_w * order)
        )
        x_pad_l, x_pad_t = int(pad_seq[0]), int(pad_seq[2])
        grad_x = grad_x_rec[..., x_pad_t : x_pad_t + h, x_pad_l : x_pad_l + w]

        gy = grad_y_avg.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
        gx = x_sum.permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=-2)
        grad_w = (gy @ gx).sum(dim=0).unsqueeze(-1).unsqueeze(-1)
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None, None, None, None


class PointwiseConvWHTOpFLOPsProfiling(Function):
    @staticmethod
    def flop_accurate_wht_out(x, masks):
        n_masks = len(masks)
        x_flat = x.flatten(start_dim=-2)
        n, c, ph, pw, l = x_flat.shape
        masks = (
            (masks == -1)
            .view(n_masks, 1, 1, 1, 1, l)
            .broadcast_to((n_masks, n, c, ph, pw, l))
        )
        x_broadcast = (
            x_flat.unsqueeze(0).broadcast_to((n_masks, n, c, ph, pw, l)).clone()
        )
        x_broadcast[masks] = x_broadcast[masks].neg()
        x_wht = th.tensor(x_broadcast.numpy().sum(axis=5))
        return x_wht

    @staticmethod
    def flop_accurate_wht_in(x_wht, masks, order):
        n_masks, n, c, ph, pw = x_wht.shape
        masks = (
            (masks == -1)
            .view(n_masks, 1, 1, 1, 1, order, order)
            .broadcast_to((n_masks, n, c, ph, pw, order, order))
        )
        x_broadcast = (
            x_wht.unsqueeze(-1)
            .unsqueeze(-1)
            .broadcast_to((n_masks, n, c, ph, pw, order, order))
            .clone()
        )
        x_broadcast[masks] = x_broadcast[masks].neg()
        x = th.tensor(x_broadcast.numpy().sum(axis=0))
        return x

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, order, masks, no_grad_x = args
        b, _, h, w = x.shape
        c_o, c_i = weight.shape[:2]
        n_mask = masks.shape[0]
        masks = masks.view(n_mask, 1, 1, 1, 1, order, order)
        y = conv2d(x, weight=weight, bias=bias)
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_pad_t, x_pad_l = ceil((p_h * order - h) / 2), ceil((p_w * order - w) / 2)
        x_pad_b, x_pad_r = p_h * order - h - x_pad_t, p_w * order - w - x_pad_l
        pad_seq = x_pad_l, x_pad_r, x_pad_t, x_pad_b
        x_pad = pad(x, pad_seq, mode="reflect")
        x_patch = x_pad.view(b, c_i, p_h, order, p_w, order).permute(0, 1, 2, 4, 3, 5)
        x_sum = PointwiseConvWHTOpFLOPsProfiling.flop_accurate_wht_out(x_patch, masks)
        cfgs = th.tensor([bias is not None, h, w, order, no_grad_x])
        pad_seq = th.tensor(pad_seq)
        ctx.save_for_backward(x_sum, weight, cfgs, pad_seq, masks)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight, cfgs, pad_seq, masks = ctx.saved_tensors
        has_bias, h, w, order, no_grad_x = cfgs
        print(f"Pointwise {x_sum.shape} {no_grad_x} {has_bias}")
        pad_seq = [int(p) for p in pad_seq]
        (grad_y,) = grad_outputs
        n_mask, b, _, p_h, p_w = x_sum.shape
        c_o, c_i = weight.shape[:2]
        order = int(order)
        grad_y_pad = pad(grad_y, pad_seq, mode="reflect")
        grad_y_patch = grad_y_pad.view(b, c_o, p_h, order, p_w, order).permute(
            0, 1, 2, 4, 3, 5
        )
        grad_y_avg = (
            PointwiseConvWHTOpFLOPsProfiling.flop_accurate_wht_out(grad_y_patch, masks)
            / 64
        )

        if not no_grad_x:
            print(grad_y_avg.shape)
            gy = grad_y_avg.permute(0, 1, 3, 4, 2)
            gx = gy @ weight.view((c_o, c_i))
            gx = gx.permute(0, 1, 4, 2, 3)
            grad_x_rec = PointwiseConvWHTOpFLOPsProfiling.flop_accurate_wht_in(
                gx, masks, order
            )
            grad_x_rec = grad_x_rec.permute(0, 1, 2, 4, 3, 5).reshape(
                b, c_i, p_h * order, p_w * order
            )
            x_pad_l, x_pad_t = int(pad_seq[0]), int(pad_seq[2])
            grad_x = grad_x_rec[..., x_pad_t : x_pad_t + h, x_pad_l : x_pad_l + w]
        else:
            grad_x = None

        gy = grad_y_avg.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
        gx = x_sum.permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=-2)
        grad_w = (gy @ gx).sum(dim=0).unsqueeze(-1).unsqueeze(-1)

        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None, None, None, None


class PointwiseConvWHT(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        order=4,
        activate=False,
        extra_cfg=dict(),
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.order = order
        self.activate = activate
        self.extra_cfg = extra_cfg
        self.profile_flops = extra_cfg.get("profile_flops", False)
        self.no_grad_x = extra_cfg.get("no_grad_x", False)
        self.register_buffer("wht_base", hadamard_func(self.order).float())
        if "mask_pos" in extra_cfg:
            self.mask_pos = extra_cfg["mask_pos"]
            self.mask_desc_str = str(extra_cfg["mask_pos"])
        elif extra_cfg.get("lp_tri_order", -1) != -1:
            o = extra_cfg["lp_tri_order"]
            self.mask_pos = [
                (r, c) for r in range(min(8, o)) for c in range(min(8, o - r))
            ]
            self.mask_desc_str = f"lp-tri-{o}"
        else:
            self.mask_pos = [(0, 0)]
            self.mask_desc_str = "default-[0, 0]"
        self.wht_pwr_logger_alpha = extra_cfg.get("logger_alpha", -1)

    def extra_repr(self) -> str:
        ret = super().extra_repr()
        ret += f", filter={self.mask_desc_str}, activate={self.activate}"
        return ret

    @property
    def mask_pos(self):
        return self._mask_pos

    @mask_pos.setter
    def mask_pos(self, pos):
        masks = []
        walsh2hadamard = generate_order(self.order)
        walsh_pos = []
        for idx_r, idx_c in pos:
            br = self.wht_base[idx_r].view(-1, 1)
            bc = self.wht_base[idx_c].view(1, -1)
            masks.append(br @ bc)
            rr = walsh2hadamard[idx_r]
            rc = walsh2hadamard[idx_c]
            walsh_pos.append(rr * 8 + rc)
        pos_idx = th.zeros(64, dtype=th.int32) - 1
        for i, p in enumerate(walsh_pos):
            pos_idx[p] = i
        walsh_pos = th.tensor(walsh_pos)
        masks = th.stack(masks)
        if hasattr(self, "wht_masks"):
            delattr(self, "wht_masks")
        if hasattr(self, "wht_walsh_pos"):
            delattr(self, "wht_walsh_pos")
        self.register_buffer("wht_masks", masks)
        self.register_buffer("wht_walsh_pos", pos_idx)
        self._mask_pos = pos

    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.activate and self.training:
            if self.profile_flops:
                y = PointwiseConvWHTOpFLOPsProfiling.apply(
                    input,
                    self.weight,
                    self.bias,
                    self.order,
                    self.wht_masks,
                    self.no_grad_x,
                )
            else:
                y = PointwiseConvWHTOp.apply(
                    input, self.weight, self.bias, self.order, self.wht_masks
                )
        else:
            y = super().forward(input)
        return y


def wrap_pointwise_conv_wht_layer(
    conv_layer: nn.Conv2d, order, activate, extra_cfg=dict(), **kwargs
):
    new_op = PointwiseConvWHT(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
        order=order,
        activate=activate,
        extra_cfg=extra_cfg,
    )
    new_op.weight.data = conv_layer.weight.data
    if new_op.bias is not None:
        new_op.bias.data = conv_layer.bias.data
    return new_op
