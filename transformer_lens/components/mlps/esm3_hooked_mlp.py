"""Hooked Transformer MLP Component.

This module contains all the component :class:`MLP`.
"""

from typing import Dict, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.addmm import batch_addmm

#Taken from: 
#https://github.com/evolutionaryscale/esm/blob/main/esm/layers/blocks.py
#To do - add license
#To do- understand how dmlp should be used here
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)

class HookedESM3MLP(CanBeUsedAsMLP):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)
        self.select_activation_function()

        self.l1 = nn.Linear(
            in_features=self.cfg.d_model, 
            out_features=swiglu_correction_fn(self.cfg.esm3_mlp_expansion_ratio, self.cfg.d_model) * 2, 
            bias=self.cfg.esm3_bias,
            dtype=self.cfg.dtype
        )
        self.l2 = nn.Linear(
            in_features=swiglu_correction_fn(self.cfg.esm3_mlp_expansion_ratio, self.cfg.d_model),
            out_features=self.cfg.d_model,
            bias=self.cfg.esm3_bias,
            dtype=self.cfg.dtype
        )

        self.hook_pre = HookPoint()  # [batch, pos, hidden]
        self.hook_post = HookPoint()  # [batch, pos, hidden/2]

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        pre_act = self.hook_pre(self.l1(x)) 
        post_act = self.hook_post(self.act_fn(pre_act)) 
        return  self.l2(post_act)
