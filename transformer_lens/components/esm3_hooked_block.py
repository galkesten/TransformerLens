"""ESM3 Unified Transformer Block Component.
Based on this code. copyright : https://github.com/evolutionaryscale/esm/blob/main/esm/layers/blocks.py 
//To do : add license 
//To do - add geometric attention
"""

from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import (
    Attention,
    LayerNorm,
    HookedESM3MLP
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import repeat_along_head_dimension
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from esm.utils.structure.affine3d import Affine3D
from esm.layers.geom_attention import (
    GeometricReasoningOriginalImpl,
)

# Transformer Block
class HookedEsm3UnifiedTransformerBlock(nn.Module):
    ln1: nn.Module
    ln2: nn.Module
    mlp: CanBeUsedAsMLP

    def __init__(self, 
    cfg: Union[Dict, HookedTransformerConfig], 
    block_index,
    use_geom_attn=False,
    ):
        super().__init__()

        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
        self.attn = Attention(cfg, "global", block_index)
        self.mlp = HookedESM3MLP(cfg)
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if self.cfg.esm3_v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=self.cfg.d_model,
                v_heads=self.cfg.esm3_v_heads,
                bias=self.cfg.esm3_bias,
                mask_and_zero_frameless=self.cfg.esm3_mask_and_zero_frameless,
            )
            self.hook_geo_attn_in = HookPoint()
            self.hook_geo_attn_out = HookPoint()

        self.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid_geo = HookPoint()


    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        sequence_id: Optional[Int[torch.Tensor, "batch pos"]],
        frames: Optional[Affine3D],
        frames_mask: Optional[torch.Tensor],
        chain_id: Optional[torch.Tensor],
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
        Returns:
            Float[torch.Tensor, "batch pos d_model"]: Our resulting tensor
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        attn_in = resid_pre
        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )

        if self.cfg.use_split_qkv_input:
            n_heads = self.cfg.n_heads
            query_input = self.hook_q_input(
                repeat_along_head_dimension(resid_pre, n_heads=n_heads)
            )
            key_input = self.hook_k_input(
                repeat_along_head_dimension(resid_pre, n_heads=n_heads)
            )
            value_input = self.hook_v_input(
                repeat_along_head_dimension(resid_pre, n_heads=n_heads)
            )
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in

        attn_out = (
            # hook the residual stream states that are used to calculate the
            # queries, keys and values, independently.
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input=self.ln1(query_input),
                key_input=self.ln1(key_input),
                value_input=self.ln1(value_input),
                past_kv_cache_entry=None,
                attention_mask=None,
            )
        )  # [batch, pos, d_model]
        attn_out = self.hook_attn_out(attn_out)
        scaled_attn_out = attn_out / self.cfg.esm3_scaling_factor
        resid_mid = self.hook_resid_mid(resid_pre +scaled_attn_out)  # [batch, pos, d_model]

        if self.use_geom_attn:
            geo_attn_in = self.hook_geo_attn_in(resid_mid.clone())
            geo_attn_out =  self.hook_geo_attn_out(self.geom_attn(geo_attn_in, frames, frames_mask, sequence_id, chain_id))
            scaled_geo_attn = geo_attn_out/self.cfg.esm3_scaling_factor
            resid_mid = self.hook_resid_mid_geo(resid_mid + scaled_geo_attn)

        mlp_in = (
                resid_mid if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_mid.clone())
            )
        normalized_resid_mid = self.ln2(mlp_in)
        mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
        scaled_mlp = mlp_out/self.cfg.esm3_scaling_factor
        resid_post = self.hook_resid_post(resid_mid+scaled_mlp)  # [batch, pos, d_model]
        return resid_post
