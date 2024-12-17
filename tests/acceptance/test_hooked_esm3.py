import pytest
import torch
from jaxtyping import Float
from torch.testing import assert_close
import torch.nn as nn
from transformer_lens.components import Attention
from transformer_lens.components import LayerNorm
from transformer_lens.components import HookedESM3MLP, swiglu_correction_fn
from transformer_lens.components import HookedEsm3UnifiedTransformerBlock
from esm.layers.attention import MultiHeadAttention
from esm.layers.blocks import swiglu_ln_ffn, UnifiedTransformerBlock
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import functools
import einops
import math 

ATOL = 1e-05
RTOL=1e-05
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("use_attn_result", [False, True])
@pytest.mark.parametrize("qk_layernorm", [False, True])
def test_compare_esm_attention_and_pytorch_attention(bias, use_attn_result, qk_layernorm):
    d_model = 512
    n_heads = 8
    d_head = d_model // n_heads
    batch_size = 1
    seq_len = 10

    fake_params = create_multi_head_attention_params(d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm)

    # create esm original attention component
    esm_original_component = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm).to(torch.float32)

    # Assign the explicit parameters to the model
    assign_params_to_esm_attention_layer(esm_original_component, fake_params, bias)

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
    attn_only=True,
    use_attn_result=use_attn_result)

    with torch.no_grad():
        tested_attention_layer = Attention(cfg)
        pre_layer_norm = LayerNorm(cfg, d_model)
        assign_params_to_transformer_lens_attention_layer(tested_attention_layer, pre_layer_norm,fake_params, cfg, bias)

        x= torch.rand((batch_size,seq_len, d_model))
        attention_res1 = esm_original_component.forward(x.clone(), None)

        layer_norm1= pre_layer_norm(x.clone())
        attention_res2 = tested_attention_layer(layer_norm1, layer_norm1, layer_norm1)
        assert(torch.allclose(attention_res1, attention_res2, atol=ATOL, rtol=RTOL))
        diff = torch.abs(attention_res1 - attention_res2)
        print("Maximum absolute difference:", torch.max(diff))
        print("Mean absolute difference:", torch.mean(diff))


def create_multi_head_attention_params(d_model, n_heads, bias=False, qk_layernorm=False):
    params = {
        "layernorm_qkv_weight": torch.rand(d_model),  # Weight of LayerNorm
        "layernorm_qkv_bias": torch.rand(d_model),    # Bias of LayerNorm
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
    

def assign_params_to_esm_attention_layer(layer, params, bias=True):
    with torch.no_grad():
        # Assign LayerNorm for QKV
        layer.layernorm_qkv[0].weight.copy_(params["layernorm_qkv_weight"])
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
        if isinstance(pre_layer_norm, nn.LayerNorm):
            pre_layer_norm.weight.copy_(params["layernorm_qkv_weight"])
            pre_layer_norm.bias.copy_(params["layernorm_qkv_bias"])
        else:
            pre_layer_norm.w.copy_(params["layernorm_qkv_weight"])
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



def create_mlp_params(d_model, expansion_ratio, bias):
    hidden_dim = swiglu_correction_fn(expansion_ratio, d_model)
    params = {
        "layernorm_weight": torch.rand(d_model),
        "layernorm_bias": torch.rand(d_model),
        "l1_weight": torch.rand(hidden_dim * 2, d_model),
        "l1_bias": torch.rand(hidden_dim * 2) if bias else None,
        "l2_weight": torch.rand(d_model, hidden_dim),
        "l2_bias": torch.rand(d_model) if bias else None,
    }
    return params


def assign_params_to_swiglu_mlp(mdl, params, bias):
    with torch.no_grad():
        # Assign LayerNorm parameters
        mdl[0].weight.copy_(params["layernorm_weight"])
        mdl[0].bias.copy_(params["layernorm_bias"])
        # Assign first Linear layer parameters
        mdl[1].weight.copy_(params["l1_weight"])
        if bias:
            mdl[1].bias.copy_(params["l1_bias"])
        # Assign second Linear layer parameters
        mdl[3].weight.copy_(params["l2_weight"])
        if bias:
            mdl[3].bias.copy_(params["l2_bias"])

def assign_params_to_esm_mlp(mdl, params, bias, pre_layer_norm):
    with torch.no_grad():
        # Assign LayerNorm
        if isinstance(pre_layer_norm, nn.LayerNorm): 
            pre_layer_norm.weight.copy_(params["layernorm_weight"])
            pre_layer_norm.bias.copy_(params["layernorm_bias"])
        else:
            pre_layer_norm.w.copy_(params["layernorm_weight"])
            pre_layer_norm.b.copy_(params["layernorm_bias"])
        # Assign first Linear layer parameters
        mdl.l1.weight.copy_(params["l1_weight"])
        if bias:
            mdl.l1.bias.copy_(params["l1_bias"])
        # Assign second Linear layer parameters
        mdl.l2.weight.copy_(params["l2_weight"])
        if bias:
            mdl.l2.bias.copy_(params["l2_bias"])

@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("expansion_ratio", [2.0, 4.0])
def test_compare_esm_and_swiglu_mlp(bias, expansion_ratio):
    d_model = 512
    batch_size = 1
    seq_len = 10
    d_head=64

    # Create fake parameters for testing
    fake_params = create_mlp_params(d_model, expansion_ratio, bias)

    # Create the SwiGLU-based MLP
    swiglu_mlp = swiglu_ln_ffn(d_model, expansion_ratio, bias)

    # Assign parameters to SwiGLU MLP
    assign_params_to_swiglu_mlp(swiglu_mlp, fake_params, bias)

    # Create ESM3_Hooked_MLP configuration
    cfg = HookedTransformerConfig(     
        n_layers=1,      
        d_model=d_model,           
        n_ctx=20,            
        d_head=d_head,                     
        init_weights=False,
        dtype=torch.float32,
        esm3_mlp_expansion_ratio=expansion_ratio,
        act_fn = "swiglu",
        esm3_bias = bias
    )

    # Create ESM3_Hooked_MLP
    esm_mlp = HookedESM3MLP(cfg)
    pre_layer_norm = LayerNorm(cfg, d_model)
    # Assign parameters to ESM3_Hooked_MLP
    assign_params_to_esm_mlp(esm_mlp, fake_params, bias, pre_layer_norm)


    # Generate input tensor
    x = torch.rand((batch_size, seq_len, d_model))

    # Forward pass through both MLPs
    with torch.no_grad():
        original_output = swiglu_mlp(x.clone())
        layer_norm1= pre_layer_norm(x.clone())
        hooked_output = esm_mlp(layer_norm1)

    # Compare outputs
    assert torch.allclose(original_output, hooked_output, atol=ATOL, rtol=RTOL), "Outputs do not match!"
    diff = torch.abs(original_output - hooked_output)
    print("Maximum absolute difference:", torch.max(diff))
    print("Mean absolute difference:", torch.mean(diff))


def assign_params_to_hooked_esm3_transformer_block(block:HookedEsm3UnifiedTransformerBlock, attention_params, mlp_params ,bias, cfg):
    attn = block.attn
    attn_layer_norm = block.ln1
    assign_params_to_transformer_lens_attention_layer(attn, attn_layer_norm, attention_params, cfg, bias=bias)
    mlp = block.mlp
    mlp_layer_norm = block.ln2
    assign_params_to_esm_mlp(mlp, mlp_params,bias,mlp_layer_norm)


def assign_params_to_original_transformer_block(block:UnifiedTransformerBlock, attention_params, mlp_params ,bias):
    attention_layer = block.attn
    assign_params_to_esm_attention_layer(attention_layer, attention_params, bias)
    mlp = block.ffn
    assign_params_to_swiglu_mlp(mlp, mlp_params, bias)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("use_attn_in", [False, True])
@pytest.mark.parametrize("use_hook_mlp_in", [False, True])
@pytest.mark.parametrize("use_split_qkv_input", [False, True])
@pytest.mark.parametrize("residue_scaling_factor", [1.0, math.sqrt(48 / 36)])
def test_compare_unified_and_hooked_transformer_blocks(bias, residue_scaling_factor, use_attn_in, use_hook_mlp_in, use_split_qkv_input):
    d_model = 512
    n_heads = 8
    d_head = d_model // n_heads
    expansion_ratio = 8/3
    batch_size = 1
    seq_len = 10
    qk_layernorm = True

    attention_fake_params = create_multi_head_attention_params(d_model=d_model,  n_heads=n_heads, 
    qk_layernorm=qk_layernorm, bias=bias)

    mlp_fake_params = create_mlp_params(d_model=d_model, expansion_ratio=expansion_ratio, bias=bias)

    original_block:UnifiedTransformerBlock = UnifiedTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        use_geom_attn=False,
        use_plain_attn=True,
        v_heads=None,
        bias=bias,
        expansion_ratio=expansion_ratio,
        residue_scaling_factor=residue_scaling_factor,
        qk_layernorm= qk_layernorm,
        ffn_type="swiglu",
    )
    
    assign_params_to_original_transformer_block(original_block, attention_fake_params, mlp_fake_params, bias)
    # Initialize HookedEsm3UnifiedTransformerBlock
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
    use_attn_result=False, 
    esm3_mlp_expansion_ratio=expansion_ratio,
    act_fn = "swiglu",
    esm3_bias = bias,
    use_attn_in = use_attn_in,
    use_hook_mlp_in = use_attn_in,
    use_split_qkv_input= use_attn_in,
    esm3_scaling_factor=residue_scaling_factor
)
    hooked_block:HookedEsm3UnifiedTransformerBlock = HookedEsm3UnifiedTransformerBlock(cfg, block_index=0)
    assign_params_to_hooked_esm3_transformer_block(hooked_block,attention_fake_params, mlp_fake_params, bias, cfg)

    # Input tensor
    x = torch.rand((batch_size, seq_len, d_model))

    # Forward pass
    with torch.no_grad():
        original_output =  original_block.forward(x.clone(), None, None, None, None)
        hooked_output = hooked_block.forward(x.clone(), None, None, None, None)

    # Compare outputs
    assert torch.allclose(original_output, hooked_output, atol=ATOL, rtol=1e-4), "Outputs do not match!"
    print("Maximum absolute difference:", torch.max(torch.abs(original_output - hooked_output)))
    print("Mean absolute difference:", torch.mean(torch.abs(original_output - hooked_output)))