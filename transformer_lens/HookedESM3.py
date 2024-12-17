"""Hooked ESM3.
All based on this code - https://github.com/evolutionaryscale/esm/blob/main/esm/models/esm3.py#L69
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal
from transformer_lens.components import (
    LayerNorm,
)
import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import HookedESM3Embed, HookedEsm3UnifiedTransformerBlock, HookedEsm3OutputHeads
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils.constants import esm3 as C
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from esm.utils.constants.models import (
    ESM3_OPEN_SMALL,
    normalize_model_name,
)
import math
from esm.tokenization import get_esm3_model_tokenizers
class SupportedESM3Config:
    def __init__(
        self,
        use_attn_result: bool = False,
        use_split_qkv_input: bool = False,
        use_hook_mlp_in: bool = False,
        use_attn_in: bool = False,
        esm3_output_type: Optional[str] = None,
    ):
        self.use_attn_result = use_attn_result
        self.use_split_qkv_input = use_split_qkv_input
        self.use_hook_mlp_in = use_hook_mlp_in
        self.use_attn_in = use_attn_in
        self.esm3_output_type = esm3_output_type


    seed: Optional[int] = None
class HookedESM3(HookedRootModule):
    """
    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`.
     There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings

        A lot of features are currentlu not supported
    """

    def __init__(
        self, 
        cfg: Union[HookedTransformerConfig, Dict], 
        tokenizers: TokenizerCollectionProtocol, 
        move_to_device=True):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedESM3.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedESM3"
        self.tokenizers = tokenizers

        #Currently ESM3 d_vocabs are const via output regression heads

        self.embed = HookedESM3Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        
        self.blocks = nn.ModuleList(
            [
                HookedEsm3UnifiedTransformerBlock(
                    cfg=cfg,
                    use_geom_attn=i < self.cfg.esm3_n_layers_geom,
                    block_index=i,
                )
                for i in range(self.cfg.n_layers)
            ]
        )
        self.ln_final = LayerNorm(self.cfg)
        self.unembed = HookedEsm3OutputHeads(self.cfg)

        if move_to_device:
            self.to(self.cfg.device)

        #does not support hook tokens- to do - add in the future
        self.setup()

    @overload
    def forward(
        self,
        *,
        return_type:  Literal["logits"],
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        structure_tokens: Int[torch.Tensor, "batch pos"] = None,
        ss8_tokens: Int[torch.Tensor, "batch pos"] = None,
        sasa_tokens: Int[torch.Tensor, "batch pos"] = None,
        function_tokens: Int[torch.Tensor, "batch pos dfunc"] = None,
        residue_annotation_tokens: Int[torch.Tensor, "batch pos dres"]| None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: Float[torch.Tensor, "batch pos dfunc"]| None = None,
        chain_id: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        
    ) -> Optional[Union[
        Float[torch.Tensor, "batch pos d_vocab_out"],
        ESMOutput,
    ]]:
        ...

    @overload
    def forward(
        self,
        *,
        return_type:  Literal[None],
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        structure_tokens: Int[torch.Tensor, "batch pos"] = None,
        ss8_tokens: Int[torch.Tensor, "batch pos"] = None,
        sasa_tokens: Int[torch.Tensor, "batch pos"] = None,
        function_tokens: Int[torch.Tensor, "batch pos dfunc"] = None,
        residue_annotation_tokens: Int[torch.Tensor, "batch pos dres"]| None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: Float[torch.Tensor, "batch pos dfunc"]| None = None,
        chain_id: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        
    ) -> Optional[Union[
        Float[torch.Tensor, "batch pos d_vocab_out"],
        ESMOutput,
    ]]:
        ...

    def forward(
        self,
        *,
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        structure_tokens: Int[torch.Tensor, "batch pos"] = None,
        ss8_tokens: Int[torch.Tensor, "batch pos"] = None,
        sasa_tokens: Int[torch.Tensor, "batch pos"] = None,
        function_tokens: Int[torch.Tensor, "batch pos dfunc"] = None,
        residue_annotation_tokens: Int[torch.Tensor, "batch pos dres"]| None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: Float[torch.Tensor, "batch pos dfunc"]| None = None,
        chain_id: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        return_type: Optional[str] = "logits",
    ) -> Optional[Union[
        Float[torch.Tensor, "batch pos d_vocab_out"],
        ESMOutput,
    ]]:
        """
        It's all taken from here- https://github.com/evolutionaryscale/esm/blob/main/esm/models/esm3.py#L462
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
        sequence_tokens (torch.Tensor, optional): The amino acid tokens.
        structure_tokens (torch.Tensor, optional): The structure tokens.
        ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
        sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
        function_tokens (torch.Tensor, optional): The function tokens.
        residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
        average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
        per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
            otherwise, use average_plddt.
        structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
        chain_id (torch.Tensor, optional): The chain ID
        sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
        ESMOutput: The output of the ESM3 model.

        Raises:
        ValueError: If at least one of the inputs is None.

        """
         # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        if device !=self.cfg.device:
            device = self.cfg.device
            sequence_tokens = sequence_tokens.to(device) if sequence_tokens is not None else sequence_tokens
            structure_tokens = structure_tokens.to(device) if structure_tokens is not None else structure_tokens
            ss8_tokens = ss8_tokens.to(device) if ss8_tokens is not None else ss8_tokens
            sasa_tokens = sasa_tokens.to(device) if sasa_tokens is not None else sasa_tokens
            structure_coords = structure_coords.to(device) if structure_coords is not None else structure_coords
            function_tokens = function_tokens.to(device) if function_tokens is not None else function_tokens
            residue_annotation_tokens =  residue_annotation_tokens.to(device) if residue_annotation_tokens is not None else residue_annotation_tokens
            average_plddt = average_plddt.to(device) if average_plddt is not None else average_plddt
            per_res_plddt = per_res_plddt.to(device) if per_res_plddt is not None else per_res_plddt
            chain_id = chain_id.to(device) if chain_id is not None else chain_id
            sequence_id = sequence_id.to(device) if  sequence_id is not None else  sequence_id


        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )

        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        #take only 3 positions from the -2 dimension, assuming the last dimension is also 3
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        resid = self.hook_embed(
            self.embed(sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,))
        
        for block in self.blocks:
            resid  = block(resid, sequence_id, affine, affine_mask, chain_id)
        normalised = self.ln_final(resid)

        if return_type is None:
            return None

        logits = self.unembed.forward(residual=normalised, embedding=resid)
        return logits #only returning logits is supported

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Optional[Union[Float[torch.Tensor, "batch pos d_vocab_out"],ESMOutput],], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Optional[Union[Float[torch.Tensor, "batch pos d_vocab_out"],ESMOutput],], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Optional[Union[
        Float[torch.Tensor, "batch pos d_vocab_out"],
        ESMOutput],],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")

    @classmethod
    def from_pretrained(
        cls,
        esm_cfg:SupportedESM3Config,
        model_name=ESM3_OPEN_SMALL,
        device: Optional[str] = None,
        move_to_device=True,
        dtype=torch.float32,  
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. 
        Unlike HookedTransformer, this does not yet do any preprocessing on the model
        We can ony load the open version."""
        logging.warning(
            "Please notice the licsence - todo- add license"
            "Support for ESM3 in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using ESM3 for interpretability research, keep in mind that ESM3 has some significant architectural "
            "differences to Language transformers like GPT."
        )

        assert dtype in [torch.float32, torch.float64], "dtype is not supported"

        #To do- add support ib bfloat16 for gpu and figure this out 

        if model_name!=ESM3_OPEN_SMALL:
            raise ValueError(f"Model name {model_name} is not supported for esm3.")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        d_model = 1536
        n_heads=24
        d_head = d_model//n_heads
        n_layers = 48

        cfg = HookedTransformerConfig(
        n_layers=n_layers,           
        d_model=d_model,           
        n_ctx=2048,            
        d_head=d_head,                     
        n_heads=n_heads,
        act_fn = "swiglu",
        n_devices= 1,
        device=device,
        attention_dir="bidirectional",
        init_weights=False,
        positional_embedding_type="rotary",
        rotary_dim=d_head,
        rotary_base= 10000,
        default_prepend_bos=False,
        qk_layernorm=True,
        dtype=dtype,
        use_attn_result=esm_cfg.use_attn_result, 
        use_attn_in = esm_cfg.use_attn_in,
        use_hook_mlp_in =esm_cfg.use_hook_mlp_in ,
        use_split_qkv_input= esm_cfg.use_split_qkv_input,
        esm3_mlp_expansion_ratio= 8 / 3,
        esm3_bias =False,
        esm3_scaling_factor=math.sqrt(n_layers / 36),
        esm3_output_type=1,
        esm3_mask_and_zero_frameless=True,
        esm3_n_layers_geom=1,
        esm3_v_heads = 256,
    )
        # state_dict = loading.get_pretrained_state_dict(
        #     official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        # )
        tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
        model = cls(cfg, tokenizers, move_to_device=False)

        #model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedESM3")

        return model
    # @property
    # def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
    #     """
    #     Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
    #     """
    #     return self.unembed.W_U

    # @property
    # def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
    #     """
    #     Convenience to get the unembedding bias
    #     """
    #     return self.unembed.b_U

    # @property
    # def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
    #     """
    #     Convenience to get the embedding matrix
    #     """
    #     return self.embed.embed.W_E

    # @property
    # def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
    #     """
    #     Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
    #     """
    #     return self.embed.pos_embed.W_pos

    # @property
    # def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
    #     """
    #     Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
    #     """
    #     return torch.cat([self.W_E, self.W_pos], dim=0)

    # @property
    # def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
    #     """Stacks the key weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.W_K for block in self.blocks], dim=0)

    # @property
    # def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
    #     """Stacks the query weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.W_Q for block in self.blocks], dim=0)

    # @property
    # def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
    #     """Stacks the value weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.W_V for block in self.blocks], dim=0)

    # @property
    # def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
    #     """Stacks the attn output weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.W_O for block in self.blocks], dim=0)

    # @property
    # def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
    #     """Stacks the MLP input weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).mlp.W_in for block in self.blocks], dim=0)

    # @property
    # def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
    #     """Stacks the MLP output weights across all layers"""
    #     return torch.stack([cast(BertBlock, block).mlp.W_out for block in self.blocks], dim=0)

    # @property
    # def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
    #     """Stacks the key biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.b_K for block in self.blocks], dim=0)

    # @property
    # def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
    #     """Stacks the query biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.b_Q for block in self.blocks], dim=0)

    # @property
    # def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
    #     """Stacks the value biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.b_V for block in self.blocks], dim=0)

    # @property
    # def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
    #     """Stacks the attn output biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).attn.b_O for block in self.blocks], dim=0)

    # @property
    # def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
    #     """Stacks the MLP input biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).mlp.b_in for block in self.blocks], dim=0)

    # @property
    # def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
    #     """Stacks the MLP output biases across all layers"""
    #     return torch.stack([cast(BertBlock, block).mlp.b_out for block in self.blocks], dim=0)

    # @property
    # def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
    #     """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
    #     Useful for visualizing attention patterns."""
    #     return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    # @property
    # def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
    #     """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
    #     return FactoredMatrix(self.W_V, self.W_O)

    # def all_head_labels(self) -> List[str]:
    #     """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
    #     return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]
