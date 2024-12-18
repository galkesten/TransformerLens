import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from esm.models.esm3 import ESM3
import torch as t
import functools
def convert_esm3_weights(esm3:ESM3, cfg: HookedTransformerConfig):
    #embedding layer

    state_dict = {
    key.replace("encoder", "embed.embed"): value
    for key, value in esm3.state_dict().items()
    if key.startswith("encoder")
    }

    for l in range(cfg.n_layers):
        block = esm3.transformer.blocks[l]
        if cfg.esm3_use_torch_layer_norm:
            state_dict[f"blocks.{l}.ln1.weight"] =  block.attn.layernorm_qkv[0].weight
            state_dict[f"blocks.{l}.ln1.bias"] =  block.attn.layernorm_qkv[0].bias
            state_dict[f"blocks.{l}.attn.q_ln.weight"] = block.attn.q_ln.weight
            state_dict[f"blocks.{l}.attn.k_ln.weight"] = block.attn.k_ln.weight
        else:
            state_dict[f"blocks.{l}.ln1.w"] =  block.attn.layernorm_qkv[0].weight
            state_dict[f"blocks.{l}.ln1.b"] =  block.attn.layernorm_qkv[0].bias
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
       
        out_proj =  block.attn.out_proj.weight.T  # Shape: (d_model, d_model)
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(out_proj, "(n_head d_head) d_model -> n_head d_head d_model", n_head=cfg.n_heads)

        if block.use_geom_attn:
            state_dict[f"blocks.{l}.geom_attn.distance_scale_per_head"] =  block.geom_attn.distance_scale_per_head
            state_dict[f"blocks.{l}.geom_attn.rotation_scale_per_head"] =  block.geom_attn.rotation_scale_per_head
            state_dict[f"blocks.{l}.geom_attn.s_norm.weight"] = block.geom_attn.s_norm.weight
            state_dict[f"blocks.{l}.geom_attn.proj.weight"] = block.geom_attn.proj.weight
            state_dict[f"blocks.{l}.geom_attn.out_proj.weight"] = block.geom_attn.out_proj.weight

        if cfg.esm3_use_torch_layer_norm:
            state_dict[f"blocks.{l}.ln2.weight"] =  block.ffn[0].weight
            state_dict[f"blocks.{l}.ln2.bias"] =   block.ffn[0].bias
        else:
            state_dict[f"blocks.{l}.ln2.w"] =  block.ffn[0].weight
            state_dict[f"blocks.{l}.ln2.b"] =   block.ffn[0].bias

        state_dict[f"blocks.{l}.mlp.l1.weight"] = block.ffn[1].weight
        state_dict[f"blocks.{l}.mlp.l2.weight"] = block.ffn[3].weight

    if cfg.esm3_use_torch_layer_norm:
        state_dict["ln_final.weight"]= esm3.transformer.norm.weight
    else:
        state_dict["ln_final.w"]= esm3.transformer.norm.weight
    output_heads_dict = {
    key.replace("output_heads", "unembed.output_heads"): value
    for key, value in esm3.state_dict().items()
    if key.startswith("output_heads")
    }
    state_dict.update(output_heads_dict)
    return state_dict
