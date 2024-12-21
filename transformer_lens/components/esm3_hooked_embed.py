"""Hooked Transformer ESM3 Hooked Embed Component.
//Taken from here : https://github.com/evolutionaryscale/esm/blob/main/esm/models/esm3.py#L69
//To do - add license
This module contains all the component :class:`ESM3HookedEmbed`.
"""
from typing import Dict, Optional, Union

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import Embed, LayerNorm, PosEmbed, TokenTypeEmbed
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from esm.models.esm3 import EncodeInputs

BatchOr1PosInt = Union[Int[torch.Tensor, "batch pos"], Int[torch.Tensor, "1 pos"]]
BatchOr1PosFloat = Union[Float[torch.Tensor, "batch pos"], Float[torch.Tensor, "1 pos"]]
BatchOr1PosD1 = Union[Int[torch.Tensor, "batch pos d1"], Int[torch.Tensor, "1 pos d1"]]
BatchOr1PosD2 = Union[Int[torch.Tensor, "batch pos d2"], Int[torch.Tensor, "1 pos d2"]]

class HookedESM3Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.embed =EncodeInputs(self.cfg.d_model)

    def forward(
        self,
        sequence_tokens: BatchOr1PosInt,
        structure_tokens: BatchOr1PosInt,
        average_plddt: BatchOr1PosFloat,
        per_res_plddt: BatchOr1PosFloat,
        ss8_tokens: BatchOr1PosInt,
        sasa_tokens: BatchOr1PosInt,
        function_tokens: BatchOr1PosD1,
        residue_annotation_tokens: BatchOr1PosD2,
    )-> Float[torch.Tensor, "batch pos d_model"]:
       return self.embed.forward(sequence_tokens=sequence_tokens, structure_tokens=structure_tokens,
       average_plddt=average_plddt, per_res_plddt=per_res_plddt,
       sasa_tokens=sasa_tokens, ss8_tokens=ss8_tokens, function_tokens=function_tokens, residue_annotation_tokens=residue_annotation_tokens)