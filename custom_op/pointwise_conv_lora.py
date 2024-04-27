"""
The code is copied from LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layer import LoRALayer


class PointwiseConvLORA(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        activate: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels,
                           kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        self.activate = activate

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @
                                     self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @
                                     self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.activate:
            if self.r > 0 and not self.merged:
                result = F.conv2d(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
                result_lora = F.conv2d(x, self.lora_A.view(
                    self.r, self.in_channels, *self.kernel_size), None, self.stride, self.padding, self.dilation, self.groups)
                result_lora = F.conv2d(result_lora, self.lora_B.view(
                    self.out_channels, self.r, *self.kernel_size), None, self.stride, self.padding, self.dilation, self.groups)
                return result + result_lora
            return nn.Conv2d.forward(self, x)
        else:
            if not self.merged:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    self.weight.data += (self.lora_B @
                                         self.lora_A).view(self.weight.shape) * self.scaling
                    self.merged = True
            return nn.Conv2d.forward(self, x)


def wrap_pointwise_conv_lora_layer(conv_layer: nn.Conv2d, order, activate=True, extra_cfg=dict(), **kwargs):
    new_op = PointwiseConvLORA(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size[0],
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
        r=order,
        lora_alpha=extra_cfg.get("alpha", order),
        activate=activate
    )
    new_op.weight.data = conv_layer.weight.data
    if new_op.bias is not None:
        new_op.bias.data = conv_layer.bias.data
    return new_op


if __name__ == "__main__":

    conv2d = nn.Conv2d(in_channels=128, out_channels=256,
                       kernel_size=1, stride=1, bias=True)
    conv2d_lora = wrap_pointwise_conv_lora_layer(conv2d, order=8)
    print(conv2d)
    print(conv2d_lora)
    print(torch.equal(conv2d.weight.data, conv2d_lora.weight.data))
    print(torch.equal(conv2d.bias.data, conv2d_lora.bias.data))
