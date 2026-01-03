"""Hooked ESMC.
All based on this code - https://github.com/evolutionaryscale/esm/blob/main/esm/models/esmc.py
"""

from __future__ import annotations

import logging
from typing import cast, overload

import torch
from jaxtyping import Float, Int
from torch import nn
from typing_extensions import Literal
from transformer_lens.components import (
    LayerNorm,
)
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import HookedEsm3UnifiedTransformerBlock
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from esm.utils.constants.models import (
    ESMC_600M,
    ESMC_300M
)

import math
from esm.tokenization import get_esmc_model_tokenizers 
from transformer_lens.pretrained.weight_conversions import convert_esmc_weights
from esm.layers.regression_head import RegressionHead
from esm.models.esmc import ESMC

class SupportedESMCConfig:
    def __init__(
        self,
        use_attn_result: bool = False,
        use_split_qkv_input: bool = False,
        use_hook_mlp_in: bool = False,
        use_attn_in: bool = False,
        esmc_use_torch_layer_norm:bool = False,
        esmc_use_org_rotary:bool = True,
        esmc_use_torch_attention_calc:bool = False,
        esmc_capture_activations_before_normalization:bool=True
    ):
        self.use_attn_result = use_attn_result
        self.use_split_qkv_input = use_split_qkv_input
        self.use_hook_mlp_in = use_hook_mlp_in
        self.use_attn_in = use_attn_in
        self.esmc_use_torch_layer_norm = esmc_use_torch_layer_norm
        self.esmc_use_org_rotary=esmc_use_org_rotary
        self.esmc_use_torch_attention_calc=esmc_use_torch_attention_calc
        self.esmc_capture_activations_before_normalization=esmc_capture_activations_before_normalization



class HookedESMC(HookedRootModule):
    """
    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`.
     There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings

        A lot of features are currently not supported
    """

    def __init__(
        self, 
        cfg: HookedTransformerConfig | dict, 
        tokenizer: EsmSequenceTokenizer, 
        move_to_device: bool = True):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedESM3.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedESM3"
        self.tokenizer = tokenizer

        self.embed = nn.Embedding(64, self.cfg.d_model)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        
        self.blocks = nn.ModuleList(
            [
                HookedEsm3UnifiedTransformerBlock(
                    cfg=cfg,
                    use_geom_attn=False,
                    block_index=i,
                )
                for i in range(self.cfg.n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(cfg.d_model, bias=cfg.esm3_bias) if cfg.esm3_use_torch_layer_norm else LayerNorm(self.cfg)
        self.unembed = RegressionHead(d_model=self.cfg.d_model, output_dim=64)

        if move_to_device and self.cfg.device is not None:
            self.to(self.cfg.device)

        #does not support hook tokens- to do - add in the future
        self.setup()

    @overload
    def forward(
        self,
        *,
        return_type: Literal["logits"],
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"]:
        ...

    @overload
    def forward(
        self,
        *,
        return_type: Literal[None],
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        
    ) -> None:
        ...

    def forward(
        self,
        *,
        sequence_tokens: Int[torch.Tensor, "batch pos"] | None = None,
        sequence_id: Int[torch.Tensor, "batch pos"] | None = None,
        return_type: str | None = "logits",
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"] | None:
        """
        It's all taken from here-https://github.com/evolutionaryscale/esm/blob/main/esm/models/esmc.py
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
        sequence_tokens (torch.Tensor, optional): The amino acid tokens.
        sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
        Float[torch.Tensor, "batch pos d_vocab_out"]: The output of the ESMC model or None if return_type is None.

        Raises:
        ValueError: If at least one of the inputs is None.

        """

        resid = self.hook_embed(self.embed(sequence_tokens))
        
        if sequence_id is None:
            sequence_id = sequence_tokens != self.tokenizer.pad_token_id
            B, L = resid.shape[:2]
            assert sequence_id.shape == (B, L)
            sequence_id = sequence_id.to(resid.device)

        for block in self.blocks:
            resid  = block(resid, sequence_id, None, None, None)
        normalised = self.ln_final(resid)

        if return_type is None:
            return None

        logits = self.unembed.forward(normalised)
        return logits #only returning logits is supported

    @overload
    def run_with_cache(  # type: ignore[override]
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs  # type: ignore[misc]
    ) -> tuple[
            Float[torch.Tensor, "batch pos d_vocab_out"] | None,
            ActivationCache | dict[str, torch.Tensor]
        ]:
        ...

    @overload
    def run_with_cache(  # type: ignore[override]
        self, *model_args, return_cache_object: Literal[False], **kwargs  # type: ignore[misc]
    ) -> tuple[
            Float[torch.Tensor, "batch pos d_vocab_out"] | None,
            ActivationCache | dict[str, torch.Tensor]
        ]:
        ...

    def run_with_cache(  # type: ignore[override]
        self,
        *model_args,  # type: ignore[misc]
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,  # type: ignore[misc]
    ) -> tuple[
            Float[torch.Tensor, "batch pos d_vocab_out"] | None,
            ActivationCache | dict[str, torch.Tensor]
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
        device_or_dtype: torch.device | str | torch.dtype,
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):  # type: ignore[override]
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")
    
    @classmethod
    def get_state_dict(cls, device: str | torch.device | None, cfg: HookedTransformerConfig) -> dict[str, torch.Tensor]:
        if cfg.model_name == "esmc_300m":
            esmc = ESMC.from_pretrained(model_name="esmc_300m", device=device)  # type: ignore[arg-type]
        elif cfg.model_name == "esmc_600m":
            esmc = ESMC.from_pretrained(model_name="esmc_600m", device=device)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Model name {cfg.model_name} is not supported for esmc.")
        for param in esmc.parameters():
            param.requires_grad = False
        return convert_esmc_weights(esmc, cfg)

    @classmethod
    def from_pretrained(
        cls,
        esmc_cfg:SupportedESMCConfig,
        model_name: str = ESMC_600M,
        device: str | torch.device | None = None,
        move_to_device: bool = True,
        dtype: torch.dtype = torch.float32,  
    ) -> HookedESM3:

        logging.warning(
            "Please notice the licsence - todo- add license"
            "Support for ESMC in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using ESMC for interpretability research, keep in mind that ESMC has some significant architectural "
            "differences to Language transformers like GPT."
        )

        assert dtype in [torch.float32, torch.float64], "dtype is not supported"

        if model_name!=ESMC_600M and model_name!=ESMC_300M:
            raise ValueError(f"Model name {model_name} is not supported for esmc.")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name == ESMC_300M:
            d_model = 960
            n_heads= 15
            d_head = d_model//n_heads
            n_layers = 30
            model_name = "esmc_300m"
        elif model_name == ESMC_600M:
            d_model = 1152
            n_heads= 18
            d_head = d_model//n_heads
            n_layers = 36
            model_name = "esmc_600m"
        else:
            raise ValueError(f"Model name {model_name} is not supported for esmc.")
     

        cfg = HookedTransformerConfig(
            n_layers=n_layers,
            model_name=model_name,           
            d_model=d_model,           
            n_ctx=2048,            
            d_head=d_head,                     
            n_heads=n_heads,
            act_fn = "swiglu",
            n_devices= 1,
            device=device, #type: ignore[arg-type]
            attention_dir="bidirectional",
            init_weights=False,
            positional_embedding_type="rotary",
            rotary_dim=d_head,
            rotary_base= 10000,
            default_prepend_bos=False,
            qk_layernorm=True,
            dtype=dtype,
            use_attn_result=esmc_cfg.use_attn_result, 
            use_attn_in = esmc_cfg.use_attn_in,
            use_hook_mlp_in =esmc_cfg.use_hook_mlp_in ,
            use_split_qkv_input= esmc_cfg.use_split_qkv_input,
            esm3_mlp_expansion_ratio= 8 / 3,
            esm3_bias =False,
            esm3_scaling_factor=math.sqrt(n_layers / 36),
            esm3_output_type=None, #non releavent for esmc
            esm3_mask_and_zero_frameless=None, #non releavent for esmc
            esm3_n_layers_geom=0,
            esm3_v_heads = None, #non releavent for esmc
            esm3_use_torch_layer_norm= esmc_cfg.esmc_use_torch_layer_norm,
            esm3_use_org_rotary=esmc_cfg.esmc_use_org_rotary,
            esm3_use_torch_attention_calc=esmc_cfg.esmc_use_torch_attention_calc,
            esm3_capture_activations_before_normalization=esmc_cfg.esmc_capture_activations_before_normalization
        )
        
        state_dict = cls.get_state_dict(device, cfg)
        tokenizer = get_esmc_model_tokenizers()
        model = cls(cfg, tokenizer, move_to_device=False)
        model.load_state_dict(state_dict, strict=False)
        if move_to_device and cfg.device is not None:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedESMC")

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

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp1"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).mlp.l1.weight for block in self.blocks], dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_model d_mlp2"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).mlp.l2.weight for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).attn.b_O for block in self.blocks], dim=0)

    # @property
    # def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
    #     """Stacks the MLP input biases across all layers"""
    #     return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).mlp.b_in for block in self.blocks], dim=0)

    # @property
    # def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
    #     """Stacks the MLP output biases across all layers"""
    #     return torch.stack([cast(HookedEsm3UnifiedTransformerBlock, block).mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> list[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]
