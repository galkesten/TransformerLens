"""Hooked Transformer ESM3 Hooked Embed Component.
//Taken from here : https://github.com/evolutionaryscale/esm/blob/main/esm/models/esm3.py#L69
//To do - add license
This module contains all the component :class:`ESM3HookedEmbed`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from esm.models.esm3 import OutputHeads, ESMOutput
import einops

class HookedEsm3OutputHeads(nn.Module):
    
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)

        self.output_heads = OutputHeads(self.cfg.d_model)


    def forward(
        self, residual: Float[torch.Tensor, "batch pos d_model"],
        embedding: Float[torch.Tensor, "batch pos d_model"],
        ) -> Union[
        Float[torch.Tensor, "batch pos d_vocab_out"],
        Float[torch.Tensor, "batch pos d1 d2"],
        ESMOutput,
        None
    ]:
        if self.cfg.esm3_output_type is None:       
            raise ValueError("Missing esm3_output_type in config")
        if self.cfg.esm3_output_type == "all":       
            res = self.output_heads(residual, embedding)
            return res
        elif  self.cfg.esm3_output_type=="sequence":
            return self.output_heads.sequence_head(residual)
        elif  self.cfg.esm3_output_type=="structure":
            return self.output_heads.structure_head(residual)
        elif  self.cfg.esm3_output_type=="secondary_structure":
            return self.output_heads.ss8_head(residual)
        elif  self.cfg.esm3_output_type=="sasa":
            return self.output_heads.sasa_head(residual)
        elif self.cfg.esm3_output_type=="function":
            function_logits =self.output_heads.function_head(residual)
            return einops.rearrange(function_logits, "... (k v) -> ... k v", k=8)
        elif self.cfg.esm3_output_type=="residue":
            return self.output_heads.residue_head(residual)
        else:
            raise ValueError(f"Unsupported esm3_output_type: {self.cfg.esm3_output_type}")