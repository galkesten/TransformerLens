import pytest
import torch
from jaxtyping import Float
from torch.testing import assert_close
import torch.nn as nn
from transformer_lens.components import Attention
from transformer_lens.components import LayerNorm
from esm.layers.attention import MultiHeadAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import functools
import einops

def test_compare_esm_attention_and_pytorch_attention():
    d_model = 512
    n_heads = 8
    d_head = d_model // n_heads
    bias = False
    batch_size = 1
    seq_len = 10
    qk_layernorm= True

    fake_params = create_multi_head_attention_params(d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm)

    # create esm original attention component
    esm_original_component = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm).to(torch.float32)

    # Assign the explicit parameters to the model
    assign_params_to__esm_attention_layer(esm_original_component, fake_params, bias)

    #Now we want to create attention of transformer lens for comparing...

    cfg = HookedTransformerConfig(
    n_layers=1,           
    d_model=d_model,           
    n_ctx=20,            
    d_head=d_head,                     
    n_heads=n_heads,
    attention_dir="bidirectional",
    init_weights=False,
    positional_embedding_type="rotary",
    rotary_dim=d_head,
    default_prepend_bos=False,
    qk_layernorm=qk_layernorm,
    dtype=torch.float32,
    attn_only=True
)

#create transformer lens attention and initialize: 
    tested_attention_layer = Attention(cfg)
    pre_layer_norm = LayerNorm(cfg, d_model)
    assign_params_to_transformer_lens_attention_layer(tested_attention_layer, pre_layer_norm,fake_params, cfg, bias)

    x= torch.rand((batch_size,seq_len, d_model))
    print("hi")
    print(esm_original_component.forward)
    attention_res1 = esm_original_component.forward(x.clone(), None)

    #tested:
    attention_res2= pre_layer_norm(x.clone())
    attention_res2 = tested_attention_layer(attention_res2, attention_res2, attention_res2)

    print(attention_res1)
    print(attention_res2)

def create_multi_head_attention_params(d_model, n_heads, bias=False, qk_layernorm=False):
    params = {
        "layernorm_qkv_weight": torch.rand(d_model),  # Weight of LayerNorm
        "layernorm_qkv_bias": torch.rand(d_model) if bias else None,    # Bias of LayerNorm
        "W_qkv_weight": torch.rand(d_model * 3, d_model),  # Weight of Linear layer
        "W_qkv_bias": torch.rand(d_model * 3) if bias else None,  # Bias of Linear layer
        "out_proj_weight": torch.rand(d_model, d_model),  # Output projection weight
        "out_proj_bias": torch.rand(d_model) if bias else None,  # Output projection bias
    }
    
    if qk_layernorm:
        params.update({
            "q_ln_weight": torch.rand(d_model),
            "q_ln_bias": torch.rand(d_model) if bias else None,
            "k_ln_weight": torch.rand(d_model),
            "k_ln_bias": torch.rand(d_model) if bias else None,
        })
    return params
    

def assign_params_to__esm_attention_layer(layer, params, bias=True):
    with torch.no_grad():
        # Assign LayerNorm for QKV
        layer.layernorm_qkv[0].weight.copy_(params["layernorm_qkv_weight"])
        if bias:
            layer.layernorm_qkv[0].bias.copy_(params["layernorm_qkv_bias"])
        
        # Assign Weights and Bias for QKV Projection
        layer.layernorm_qkv[1].weight.copy_(params["W_qkv_weight"])
        if bias:
            layer.layernorm_qkv[1].bias.copy_(params["W_qkv_bias"])
        
        # Assign Output Projection
        layer.out_proj.weight.copy_(params["out_proj_weight"])
        if bias:
            layer.out_proj.bias.copy_(params["out_proj_bias"])
        
        # Assign LayerNorm for Q
        if isinstance(layer.q_ln, nn.LayerNorm):
            layer.q_ln.weight.copy_(params["q_ln_weight"])
            if bias:
                layer.q_ln.bias.copy_(params["q_ln_bias"])
        
        # Assign LayerNorm for K
        if isinstance(layer.k_ln, nn.LayerNorm):
            layer.k_ln.weight.copy_(params["k_ln_weight"])
            if bias:
                layer.k_ln.bias.copy_(params["k_ln_bias"])

def assign_params_to_transformer_lens_attention_layer(attention_layer, pre_layer_norm, params, cfg, bias=True):
    with torch.no_grad():
        # Assign LayerNorm QKV
        pre_layer_norm.w.copy_(params["layernorm_qkv_weight"])
        if bias and "layernorm_qkv_bias" in params:
            pre_layer_norm.b.copy_(params["layernorm_qkv_bias"])

        # Extract and split QKV weights
        qkv_matrix = params["W_qkv_weight"].clone()  # Shape: (d_model * 3, d_model)
        assert qkv_matrix.shape == (cfg.d_model * 3, cfg.d_model), "QKV weight shape mismatch."

        qkv_reshaped = qkv_matrix.T  # Shape: (d_model, d_model * 3)
        q, k, v = torch.chunk(qkv_reshaped, 3, dim=-1)  # Split into Q, K, V
        
        reshaper = functools.partial(
            einops.rearrange, pattern="d_model (n_head d_head) -> n_head d_model d_head", n_head=cfg.n_heads
        )
        q, k, v = map(reshaper, (q, k, v))
        
        # Copy Q, K, V weights
        attention_layer.W_Q.copy_(q)
        attention_layer.W_K.copy_(k)
        attention_layer.W_V.copy_(v)

        # Handle QKV bias
        if bias and "W_qkv_bias" in params:
            qkv_bias = params["W_qkv_bias"].clone()  # Shape: (d_model * 3)
            b_q, b_k, b_v = torch.chunk(qkv_bias, 3, dim=-1)
            reshaper_bias = functools.partial(
                einops.rearrange, pattern="(n_head d_head) -> n_head d_head", n_head=cfg.n_heads
            )
            attention_layer.b_Q.copy_(reshaper_bias(b_q))
            attention_layer.b_K.copy_(reshaper_bias(b_k))
            attention_layer.b_V.copy_(reshaper_bias(b_v))

        # Assign Output Projection
        out_proj = params["out_proj_weight"].clone()  # Shape: (d_model, d_model)
        assert out_proj.shape == (cfg.d_model, cfg.d_model), "Output projection weight shape mismatch."
        out_proj_reshaped = einops.rearrange(out_proj.T, "(n_head d_head) d_model -> n_head d_head d_model", n_head=cfg.n_heads)
        attention_layer.W_O.copy_(out_proj_reshaped)

        # Assign Output Bias
        if bias and "out_proj_bias" in params:
            attention_layer.b_O.copy_(params["out_proj_bias"])

        # Assign LayerNorms for Q and K if qk_layernorm is enabled
        if cfg.qk_layernorm:
            attention_layer.q_ln.w.copy_(params["q_ln_weight"])
            attention_layer.k_ln.w.copy_(params["k_ln_weight"])
            if bias:
                attention_layer.q_ln.b.copy_(params.get("q_ln_bias", torch.zeros(cfg.d_model)))
                attention_layer.k_ln.b.copy_(params.get("k_ln_bias", torch.zeros(cfg.d_model)))


if __name__ == '__main__':
    test_compare_esm_attention_and_pytorch_attention()
