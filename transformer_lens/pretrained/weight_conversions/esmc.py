import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from esm.models.esmc import ESMC
import torch as t
import functools

def convert_esmc_weights(esmc: ESMC, cfg: HookedTransformerConfig):
    #embedding layer - ESMC uses simple Embedding(64, 1152)
    state_dict = {
        key: value
        for key, value in esmc.state_dict().items()
        if key.startswith("embed")
    }

    for l in range(cfg.n_layers):
        block = esmc.transformer.blocks[l]
        if cfg.esm3_use_torch_layer_norm:
            state_dict[f"blocks.{l}.ln1.weight"] = block.attn.layernorm_qkv[0].weight
            state_dict[f"blocks.{l}.ln1.bias"] = block.attn.layernorm_qkv[0].bias
            state_dict[f"blocks.{l}.attn.q_ln.weight"] = block.attn.q_ln.weight
            state_dict[f"blocks.{l}.attn.k_ln.weight"] = block.attn.k_ln.weight
        else:
            state_dict[f"blocks.{l}.ln1.w"] = block.attn.layernorm_qkv[0].weight
            state_dict[f"blocks.{l}.ln1.b"] = block.attn.layernorm_qkv[0].bias
            state_dict[f"blocks.{l}.attn.q_ln.w"] = block.attn.q_ln.weight
            state_dict[f"blocks.{l}.attn.k_ln.w"] = block.attn.k_ln.weight
            
        # Extract and split QKV weights
        qkv_matrix = block.attn.layernorm_qkv[1].weight.T  # Shape: (d_model, d_model*3)
        q, k, v = t.chunk(qkv_matrix, 3, dim=-1)  # Split into Q, K, V
        reshaper = functools.partial(
            einops.rearrange, pattern="d_model (n_head d_head) -> n_head d_model d_head", n_head=cfg.n_heads
        )
        q, k, v = map(reshaper, (q, k, v))
        state_dict[f"blocks.{l}.attn.W_Q"] = q
        state_dict[f"blocks.{l}.attn.W_K"] = k
        state_dict[f"blocks.{l}.attn.W_V"] = v
       
        out_proj = block.attn.out_proj.weight.T  # Shape: (d_model, d_model)
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(out_proj, "(n_head d_head) d_model -> n_head d_head d_model", n_head=cfg.n_heads)

        if cfg.esm3_use_torch_layer_norm:
            state_dict[f"blocks.{l}.ln2.weight"] = block.ffn[0].weight
            state_dict[f"blocks.{l}.ln2.bias"] = block.ffn[0].bias
        else:
            state_dict[f"blocks.{l}.ln2.w"] = block.ffn[0].weight
            state_dict[f"blocks.{l}.ln2.b"] = block.ffn[0].bias

        state_dict[f"blocks.{l}.mlp.l1.weight"] = block.ffn[1].weight
        state_dict[f"blocks.{l}.mlp.l2.weight"] = block.ffn[3].weight

    if cfg.esm3_use_torch_layer_norm:
        state_dict["ln_final.weight"] = esmc.transformer.norm.weight
    else:
        state_dict["ln_final.w"] = esmc.transformer.norm.weight
    
    # sequence_head maps to unembed (RegressionHead)
    sequence_head_dict = {
        key.replace("sequence_head", "unembed"): value
        for key, value in esmc.state_dict().items()
        if key.startswith("sequence_head")
    }
    state_dict.update(sequence_head_dict)
    return state_dict

