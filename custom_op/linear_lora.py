"""
The code is copied from LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layer import LoRALayer


class LinearLORA(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        activate: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        self.activate = activate

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @
                                          self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @
                                          self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.activate:
            if self.r > 0 and not self.merged:
                result = F.linear(x, T(self.weight), bias=self.bias)
                if self.r > 0:
                    result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                               @ self.lora_B.transpose(0, 1)) * self.scaling
                return result
            else:
                return F.linear(x, T(self.weight), bias=self.bias)
        else:
            if not self.merged:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0:
                        self.weight.data += T(self.lora_B @
                                              self.lora_A) * self.scaling
                    self.merged = True
            return F.linear(x, T(self.weight), bias=self.bias)


def wrap_linear_lora_layer(linear_layer: nn.Linear, order, activate=True, extra_cfg=dict(), **kwargs):
    # TODO: activate and extra_cfg are not used now
    new_linear = LinearLORA(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        r=order,
        lora_alpha=extra_cfg.get("alpha", order),
        activate=activate
    )
    new_linear.weight.data = linear_layer.weight.data
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
    return new_linear


if __name__ == "__main__":

    linear = nn.Linear(in_features=128, out_features=256, bias=True)
    linear_lora = wrap_linear_lora_layer(linear, order=8)
    print(linear)
    print(linear_lora)
    print(torch.equal(linear.weight.data, linear_lora.weight.data))
    print(torch.equal(linear.bias.data, linear_lora.bias.data))
