"""ESM3 Unified Transformer Block Component.
Based on this code. copyright : https://github.com/evolutionaryscale/esm/blob/main/esm/layers/transformer_stack.py
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
    HookedEsm3UnifiedTransformerBlock
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from esm.utils.structure.affine3d import Affine3D
# Transformer Block
class HookedEsm3TransformerStack(nn.Module):

    def __init__(self, 
    cfg: Union[Dict, HookedTransformerConfig], 
    ):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            [
                HookedEsm3UnifiedTransformerBlock(
                    cfg=cfg,
                    use_geom_attn=i < self.cfg.esm3_n_layers_geom,
                    block_index=i,
                )
                for i in self.cfg.n_layers
            ]
        )
        self.norm = LayerNorm(cfg) #no bias
        self.pre_last_norm = HookPoint()

    def forward(
        self,
        x: Float[torch.Tensor, "batch pos d_model"],
        sequence_id: Optional[Int[torch.Tensor, "batch pos"]],
        frames: Optional[Affine3D],
        frames_mask: Optional[torch.Tensor],
        chain_id: Optional[torch.Tensor],
    ):

        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, affine, affine_mask, chain_id)
            hiddens.append(x)
        hiddens = torch.stack(hiddens, dim=0)
        normalised = self.pre_last_norm(x)
        return normalised, x, hiddens        
