import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import linear, pad
import torch.nn as nn
from math import ceil, sqrt
from .wht import hadamard_func, generate_order


class LinearWHTOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, shape, order, masks = args
        assert x.dim() == 3, f"{x.shape}"
        b = x.shape[0]
        h, w = shape
        c_o, c_i = weight.shape
        n_mask = masks.shape[0]
        masks = masks.view(n_mask, 1, 1, 1, 1, order, order)
        y = linear(x, weight, bias)
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_2d = x.view(-1, h, w, c_i).permute(0, 3, 1, 2)
        x_pad_t, x_pad_l = ceil((p_h * order - h) / 2), ceil((p_w * order - w) / 2)
        x_pad_b, x_pad_r = p_h * order - h - x_pad_t, p_w * order - w - x_pad_l
        pad_seq = x_pad_l, x_pad_r, x_pad_t, x_pad_b
        x_pad = pad(x_2d, pad_seq, mode="reflect")
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
        c_o, c_i = weight.shape
        order = int(order)
        grad_y_2d = grad_y.view(-1, h, w, c_o).permute(0, 3, 1, 2)
        grad_y_pad = pad(grad_y_2d, pad_seq, mode="reflect")
        grad_y_patch = (
            grad_y_pad.view(b, c_o, p_h, order, p_w, order)
            .permute(0, 1, 2, 4, 3, 5)
            .unsqueeze(0)
        )
        grad_y_avg = (grad_y_patch * masks).mean(dim=(-1, -2))

        grad_x_avg = (weight.t() @ grad_y_avg.flatten(start_dim=3)).view(
            n_mask, b, c_i, p_h, p_w
        )
        grad_x = grad_x_avg.view(n_mask, b, c_i, p_h, p_w, 1, 1) * masks
        grad_x_rec = (
            grad_x.sum(dim=0)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c_i, p_h * order, p_w * order)
        )
        x_pad_l, x_pad_t = int(pad_seq[0]), int(pad_seq[2])
        grad_x = grad_x_rec[..., x_pad_t : x_pad_t + h, x_pad_l : x_pad_l + w]
        grad_x = grad_x.flatten(start_dim=-2).permute(0, 2, 1)

        gy = grad_y_avg.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
        gx = x_sum.permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=-2)
        grad_w = (gy @ gx).sum(dim=0)
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 1))
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None, None, None, None


class LinearWHTFLOPsProfiling(Function):
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
        x, weight, bias, shape, order, masks, no_grad_x = args
        assert x.dim() == 3, f"{x.shape}"
        b = x.shape[0]
        h, w = shape
        c_o, c_i = weight.shape
        n_mask = masks.shape[0]
        masks = masks.view(n_mask, 1, 1, 1, 1, order, order)
        y = linear(x, weight, bias)
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_2d = x.view(-1, h, w, c_i).permute(0, 3, 1, 2)
        x_pad_t, x_pad_l = ceil((p_h * order - h) / 2), ceil((p_w * order - w) / 2)
        x_pad_b, x_pad_r = p_h * order - h - x_pad_t, p_w * order - w - x_pad_l
        pad_seq = x_pad_l, x_pad_r, x_pad_t, x_pad_b
        x_pad = pad(x_2d, pad_seq, mode="reflect")
        x_patch = x_pad.view(b, c_i, p_h, order, p_w, order).permute(0, 1, 2, 4, 3, 5)
        x_sum = LinearWHTFLOPsProfiling.flop_accurate_wht_out(x_patch, masks)
        cfgs = th.tensor([bias is not None, h, w, order, no_grad_x])
        pad_seq = th.tensor(pad_seq)
        ctx.save_for_backward(x_sum, weight, cfgs, pad_seq, masks)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight, cfgs, pad_seq, masks = ctx.saved_tensors
        has_bias, h, w, order, no_grad_x = cfgs
        print(f"ProfileBwd! {no_grad_x}")
        pad_seq = [int(p) for p in pad_seq]
        (grad_y,) = grad_outputs
        n_mask, b, _, p_h, p_w = x_sum.shape
        c_o, c_i = weight.shape
        order = int(order)
        grad_y_2d = grad_y.view(-1, h, w, c_o).permute(0, 3, 1, 2)
        grad_y_pad = pad(grad_y_2d, pad_seq, mode="reflect")
        grad_y_patch = grad_y_pad.view(b, c_o, p_h, order, p_w, order).permute(
            0, 1, 2, 4, 3, 5
        )
        grad_y_avg = (
            LinearWHTFLOPsProfiling.flop_accurate_wht_out(grad_y_patch, masks) / 64
        )

        if no_grad_x:
            grad_x = None
        else:
            grad_x_avg = (weight.t() @ grad_y_avg.flatten(start_dim=3)).view(
                n_mask, b, c_i, p_h, p_w
            )
            grad_x_rec = LinearWHTFLOPsProfiling.flop_accurate_wht_in(
                grad_x_avg, masks, order
            )
            grad_x_rec = grad_x_rec.permute(0, 1, 2, 4, 3, 5).reshape(
                b, c_i, p_h * order, p_w * order
            )
            x_pad_l, x_pad_t = int(pad_seq[0]), int(pad_seq[2])
            grad_x = grad_x_rec[..., x_pad_t : x_pad_t + h, x_pad_l : x_pad_l + w]
            grad_x = grad_x.flatten(start_dim=-2).permute(0, 2, 1)

        gy = grad_y_avg.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
        gx = x_sum.permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=-2)
        grad_w = (gy @ gx).sum(dim=0)
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 1))
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None, None, None, None


class LinearWHT(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        order=4,
        shape=None,
        activate=False,
        extra_cfg=dict(),
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.order = order
        self.activate = activate
        self.shape = shape
        self.extra_cfg = extra_cfg
        self.extra_token_num = extra_cfg.get("extra_token_num", 0)
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
        ret += f", filter={self.mask_desc_str}, active={self.activate}"
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
        masks = th.stack(masks)
        if hasattr(self, "wht_masks"):
            delattr(self, "wht_masks")
        if hasattr(self, "wht_walsh_pos"):
            delattr(self, "wht_walsh_pos")
        self.register_buffer("wht_masks", masks)
        self.register_buffer("wht_walsh_pos", pos_idx)
        self._mask_pos = pos

    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.shape is None:
            if input.dim() == 4:
                self.shape = input.shape[1:3]
            elif input.dim() == 3:
                s = int(sqrt(input.shape[1]))
                self.shape = [s, s]
        flatten = input.dim() == 4
        if flatten:
            input = input.flatten(start_dim=1, end_dim=3)
        if self.extra_token_num != 0:
            extra_token = input[:, : self.extra_token_num, :]
            y_extra = super().forward(extra_token)
            input = input[:, self.extra_token_num :, :]
        if self.activate and self.training:
            if self.profile_flops:
                y = LinearWHTFLOPsProfiling.apply(
                    input,
                    self.weight,
                    self.bias,
                    self.shape,
                    self.order,
                    self.wht_masks,
                    self.no_grad_x,
                )
            else:
                y = LinearWHTOp.apply(
                    input,
                    self.weight,
                    self.bias,
                    self.shape,
                    self.order,
                    self.wht_masks,
                )
        else:
            y = super().forward(input)
        if self.extra_token_num != 0:
            y = th.cat((y_extra, y), dim=1)
        if flatten:
            y = y.reshape(y.shape[0], *self.shape, y.shape[-1])
        return y


def wrap_linear_wht_layer(
    linear_layer: nn.Linear, order, activate, extra_cfg=dict(), **kwargs
):
    new_linear = LinearWHT(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        order=order,
        activate=activate,
        extra_cfg=extra_cfg,
    )
    new_linear.weight.data = linear_layer.weight.data
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
    return new_linear
