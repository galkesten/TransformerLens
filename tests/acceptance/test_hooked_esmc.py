import pytest
import torch
from typing import Any
from transformer_lens.components import HookedEsm3UnifiedTransformerBlock
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import einops
from transformer_lens import HookedESMC, SupportedESMCConfig
from esm.tokenization import get_esmc_model_tokenizers
import gc
from esm.pretrained import ESMC_600M_202412, ESMC_300M_202412
ATOL = 1e-05
RTOL = 1e-05

@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def verify_identical_components(real: Any, hooked: Any) -> bool:
    # Compare parameters
    for (name1, param1), (name2, param2) in zip(real.named_parameters(), hooked.named_parameters()):
        if name1 != name2:
            print(f"Mismatch in parameter names: {name1} != {name2}")
            return False
        assert torch.sum(param1 != param2) == 0

    print(f"verify_identical_components- All parameters match! {type(real)} {type(hooked)} ")
    return True

def compare_transformer_blocks(real_block: Any, hooked_block: HookedEsm3UnifiedTransformerBlock, cfg: HookedTransformerConfig) -> None:
    if cfg.esm3_use_torch_layer_norm:
        assert torch.sum(real_block.attn.layernorm_qkv[0].weight != hooked_block.ln1.weight) == 0
        assert torch.sum(real_block.attn.layernorm_qkv[0].bias != hooked_block.ln1.bias) == 0
    else:
        assert torch.sum(real_block.attn.layernorm_qkv[0].weight != hooked_block.ln1.w) == 0
        assert torch.sum(real_block.attn.layernorm_qkv[0].bias != hooked_block.ln1.b) == 0
        
    qkv_matrix = real_block.attn.layernorm_qkv[1].weight
    query_BLD, key_BLD, value_BLD = torch.chunk(qkv_matrix, 3, dim=-2)
    q = einops.rearrange(hooked_block.attn.W_Q, "n_head d_model d_head ->(n_head d_head) d_model", n_head=hooked_block.attn.W_Q.shape[0])
    v = einops.rearrange(hooked_block.attn.W_V, "n_head d_model d_head ->(n_head d_head) d_model", n_head=hooked_block.attn.W_V.shape[0])
    k = einops.rearrange(hooked_block.attn.W_K, "n_head d_model d_head ->(n_head d_head) d_model", n_head=hooked_block.attn.W_K.shape[0])
    assert torch.sum(query_BLD != q) == 0
    assert torch.sum(key_BLD != k) == 0
    assert torch.sum(value_BLD != v) == 0
    assert(real_block.attn.layernorm_qkv[1].bias is None)
    assert torch.equal(hooked_block.attn.b_Q, torch.zeros_like(hooked_block.attn.b_Q)), "The tensor is not all zeros."
    assert torch.equal(hooked_block.attn.b_K, torch.zeros_like(hooked_block.attn.b_K)), "The tensor is not all zeros."
    assert torch.equal(hooked_block.attn.b_V, torch.zeros_like(hooked_block.attn.b_V)), "The tensor is not all zeros."
    
    if cfg.esm3_use_torch_layer_norm:
        assert torch.sum(real_block.attn.q_ln.weight != hooked_block.attn.q_ln.weight) == 0
        assert torch.sum(real_block.attn.k_ln.weight != hooked_block.attn.k_ln.weight) == 0
        assert real_block.attn.q_ln.bias is None
        assert hooked_block.attn.q_ln.bias is None
        assert real_block.attn.k_ln.bias is None
        assert hooked_block.attn.k_ln.bias is None
    else:
        assert torch.sum(real_block.attn.q_ln.weight != hooked_block.attn.q_ln.w) == 0
        assert torch.sum(real_block.attn.k_ln.weight != hooked_block.attn.k_ln.w) == 0
        assert real_block.attn.q_ln.bias is None
        assert torch.equal(hooked_block.attn.q_ln.b, torch.zeros_like(hooked_block.attn.q_ln.b)), "The tensor is not all zeros."
        assert real_block.attn.k_ln.bias is None
        assert torch.equal(hooked_block.attn.k_ln.b, torch.zeros_like(hooked_block.attn.k_ln.b)), "The tensor is not all zeros."

    out_proj = real_block.attn.out_proj.weight
    W_O = einops.rearrange(hooked_block.attn.W_O, "n_head d_head d_model -> d_model (n_head d_head)", n_head=hooked_block.attn.W_O.shape[0])
    assert torch.sum(W_O != out_proj) == 0
    assert real_block.attn.out_proj.bias is None
    assert torch.equal(hooked_block.attn.b_O, torch.zeros_like(hooked_block.attn.b_O)), "The tensor is not all zeros."

    assert real_block.use_geom_attn == hooked_block.use_geom_attn
    
    if cfg.esm3_use_torch_layer_norm:
        assert torch.sum(real_block.ffn[0].weight != hooked_block.ln2.weight) == 0
        assert torch.sum(real_block.ffn[0].bias != hooked_block.ln2.bias) == 0
    else:
        assert torch.sum(real_block.ffn[0].weight != hooked_block.ln2.w) == 0
        assert torch.sum(real_block.ffn[0].bias != hooked_block.ln2.b) == 0
    assert torch.sum(real_block.ffn[1].weight != hooked_block.mlp.l1.weight) == 0
    assert(real_block.ffn[1].bias is None)
    assert(hooked_block.mlp.l1.bias is None)
    assert torch.sum(real_block.ffn[3].weight != hooked_block.mlp.l2.weight) == 0
    assert(real_block.ffn[3].bias is None)
    assert(hooked_block.mlp.l2.bias is None)
    print("compare_transformer_blocks- all params match")

@pytest.mark.parametrize("esmc_use_torch_layer_norm", [False, True])
@pytest.mark.parametrize("model_name", ["esmc_300m", "esmc_600m"])
def test_loading(device: str, esmc_use_torch_layer_norm: bool, model_name: str) -> None:
    print(f"test_loading- device: {device}, esmc_use_torch_layer_norm: {esmc_use_torch_layer_norm}, model_name: {model_name}")
    config = SupportedESMCConfig(
        use_attn_result=False,
        use_split_qkv_input=False,
        use_hook_mlp_in=True,
        use_attn_in=False,
        esmc_use_torch_layer_norm=esmc_use_torch_layer_norm,
        esmc_use_torch_attention_calc=True
    )
    esmc_hooked = HookedESMC.from_pretrained(esmc_cfg=config, model_name=model_name, device=device)
    esmc_original = ESMC_600M_202412(device=device) if model_name == "esmc_600m" else ESMC_300M_202412(device=device)
    esmc_hooked.eval()
    esmc_original.eval()
    cfg = esmc_hooked.cfg
    
    # Compare embedding
    verify_identical_components(esmc_original.embed, esmc_hooked.embed)
    
    # Compare transformer blocks
    for l in range(len(esmc_original.transformer.blocks)):
        real_block = esmc_original.transformer.blocks[l]
        hooked_block = esmc_hooked.blocks[l]
        compare_transformer_blocks(real_block, hooked_block, cfg)  # type: ignore[arg-type]
    
    # Compare final layer norm
    if cfg.esm3_use_torch_layer_norm:
        assert torch.sum(esmc_original.transformer.norm.weight != esmc_hooked.ln_final.weight) == 0
        assert esmc_hooked.ln_final.bias is None
        assert esmc_original.transformer.norm.bias is None
    else:
        assert torch.sum(esmc_original.transformer.norm.weight != esmc_hooked.ln_final.w) == 0
        assert torch.equal(esmc_hooked.ln_final.b, torch.zeros_like(esmc_hooked.ln_final.b)), "The tensor is not all zeros."
        assert esmc_original.transformer.norm.bias is None
    
    # Compare unembed (sequence_head)
    verify_identical_components(esmc_original.sequence_head, esmc_hooked.unembed)
    
    del esmc_hooked
    del esmc_original
    torch.cuda.empty_cache()


@pytest.mark.parametrize("use_attn_result", [True, False])
@pytest.mark.parametrize("use_split_qkv_input", [True, False])
@pytest.mark.parametrize("esmc_use_torch_layer_norm", [True, False])
@pytest.mark.parametrize("esmc_use_torch_attention_calc", [True, False])
@pytest.mark.parametrize("esmc_use_org_rotary", [True, False])
@pytest.mark.parametrize("esmc_capture_activations_before_normalization", [False, True])
@pytest.mark.parametrize("model_name", ["esmc_300m", "esmc_600m"])
#@pytest.mark.parametrize("model_name", ["esmc_600m"])
def test_full_model(
    device: str,
    esmc_use_torch_attention_calc: bool,
    use_attn_result: bool,
    use_split_qkv_input: bool,
    esmc_use_org_rotary: bool,
    esmc_use_torch_layer_norm: bool,
    esmc_capture_activations_before_normalization: bool,
    model_name: str,
) -> None:
    esmc_original = ESMC_600M_202412(device=device) if model_name == "esmc_600m" else ESMC_300M_202412(device=device)
    esmc_original = esmc_original.to(device).to(torch.float32).eval()
    esmc_original.eval()
    tokenizer = get_esmc_model_tokenizers()
    sequence = "MKSLLLLSILAALAVAALCYESHESLESYEINPFINRRNANSFISPQQRWRAKAQERIRELNKPQYELNREACDDFKLCERYAMVYGYNAAYDRYFRQRRGAK"
    # EsmSequenceTokenizer uses encode method
    tokens = tokenizer.encode(sequence)
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64).to(device).unsqueeze(0)
    
    with torch.no_grad():
        output1 = esmc_original.forward(
            sequence_tokens=sequence_tokens
        )
    del esmc_original
    torch.cuda.empty_cache()
    gc.collect()

    config = SupportedESMCConfig(
        use_attn_result=use_attn_result,
        use_split_qkv_input=use_split_qkv_input,
        use_hook_mlp_in=False,
        use_attn_in=False,
        esmc_use_torch_layer_norm=esmc_use_torch_layer_norm,
        esmc_use_torch_attention_calc=esmc_use_torch_attention_calc,
        esmc_use_org_rotary=esmc_use_org_rotary,
        esmc_capture_activations_before_normalization=esmc_capture_activations_before_normalization
    )
    esmc_hooked = HookedESMC.from_pretrained(esmc_cfg=config, model_name=model_name, device=device)
    esmc_hooked.eval()
    with torch.no_grad():
        output2 = esmc_hooked.forward(
            sequence_tokens=sequence_tokens,
            return_type="logits"
        )

    # ESMC only outputs sequence_logits
    assert torch.allclose(output1.sequence_logits, output2,  rtol=1e-5, atol=1e-4)

    del esmc_hooked
    torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.parametrize("esmc_use_torch_attention_calc", [True, False])
def test_attention_mask(
    device: str,
    esmc_use_torch_attention_calc: bool,
) -> None:
    tokenizer = get_esmc_model_tokenizers()
    sequence1 = "MKSLLLLSILAALAVAALCYESHESLESYEINPFINRRNANSFISPQQRWRAKAQERIRELNKPQYELNREACDDFKLCERYAMVYGYNAAYDRYFRQRRGAK"
    sequence2 = "MKTLLLTLLVVTIVCLDLGYTLECHNQQSSQTPTTTGCSGGETNCYKKRWRDHRGYRTERGCGCPSVKNGIEINCCTTDRCNN"
    tokenizer_res = tokenizer([sequence1, sequence2], return_tensors="pt", padding=True)
    sequence_tokens = tokenizer_res['input_ids'].to(device)  # type: ignore[attr-defined]
    sequence_id = tokenizer_res['attention_mask'].to(device)  # type: ignore[attr-defined]

    device_obj = torch.device(device)
    model_name = "esmc_600m"
    esmc_original = ESMC_600M_202412(device=device) if model_name == "esmc_600m" else ESMC_300M_202412(device=device)
    esmc_original = esmc_original.to(device).to(torch.float32).eval()
    esmc_original.eval()

    with torch.no_grad():
        output1 = esmc_original.forward(
            sequence_tokens=sequence_tokens, sequence_id=sequence_id
        )
    del esmc_original
    torch.cuda.empty_cache()
    gc.collect()

    config = SupportedESMCConfig(
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_mlp_in=True,
        use_attn_in=False,
        esmc_use_torch_layer_norm=True,
        esmc_use_torch_attention_calc=esmc_use_torch_attention_calc,
        esmc_use_org_rotary=True
    )
    esmc_hooked = HookedESMC.from_pretrained(esmc_cfg=config, model_name="esmc_600m", device=device)
    esmc_hooked.eval()
    with torch.no_grad():
        output2 = esmc_hooked.forward(
            sequence_tokens=sequence_tokens, 
            sequence_id=sequence_id,
            return_type="logits"
        )
    #print the absolute difference between the two outputs
    print(f"The absolute difference between the two outputs is: {torch.max(torch.abs(output1.sequence_logits - output2))}")
    assert torch.allclose(output1.sequence_logits, output2, rtol=1e-5, atol=1e-4)

    del esmc_hooked
    torch.cuda.empty_cache()
    gc.collect()


def test_masked_loss(device: str) -> None:
    device_obj = torch.device(device)
    model_name = "esmc_600m"
    esmc_original = ESMC_600M_202412(device=device) if model_name == "esmc_600m" else ESMC_300M_202412(device=device)
    esmc_original = esmc_original.to(device).to(torch.float32).eval()
    esmc_original.eval()
    tokenizer = get_esmc_model_tokenizers()

    # Input sequence
    sequence = "MKSLLLLSILAALAVAALCYESHESLESYEINPFINRRNANSFISPQQRWRAKAQERIRELNKPQYELNREACDDFKLCERYAMVYGYNAAYDRYFRQRRGAK"
    tokens = tokenizer.encode(sequence)
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64).to(device).unsqueeze(0)

    # Randomly mask tokens
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        pytest.skip("Tokenizer does not have mask_token_id")
    mask_prob = 0.15  # 15% of tokens will be masked
    random_mask = torch.bernoulli(torch.full(sequence_tokens.shape, mask_prob)).bool().to(device)
    masked_sequence = sequence_tokens.clone()
    masked_sequence[random_mask] = mask_token_id  # Replace with <mask>

    with torch.no_grad():
        # Get logits and calculate loss for original model
        output_original = esmc_original.forward(sequence_tokens=masked_sequence)
        logits_original = output_original.sequence_logits
        target_original = sequence_tokens[random_mask]
        loss_original = torch.nn.functional.cross_entropy(
            logits_original[random_mask], target_original
        )

    del esmc_original
    torch.cuda.empty_cache()
    gc.collect()

    config = SupportedESMCConfig(
        use_attn_result=True,
        use_split_qkv_input=True,
        use_hook_mlp_in=True,
        use_attn_in=False,
        esmc_use_torch_layer_norm=True,
        esmc_use_torch_attention_calc=True,
        esmc_use_org_rotary=True
    )
    esmc_hooked = HookedESMC.from_pretrained(esmc_cfg=config, model_name="esmc_600m", device=device)
    esmc_hooked.eval()

    with torch.no_grad():
        # Get logits and calculate loss for hooked model
        output_hooked = esmc_hooked.forward(sequence_tokens=masked_sequence, return_type="logits")
        logits_hooked = output_hooked
        target_hooked = sequence_tokens[random_mask]
        loss_hooked = torch.nn.functional.cross_entropy(
            logits_hooked[random_mask], target_hooked
        )
    print(loss_original)
    print(loss_hooked)
    print(loss_original - loss_hooked)

    # Assert losses are approximately equal
    assert torch.allclose(loss_original, loss_hooked, rtol=0.0, atol=4e-5)

    del esmc_hooked
    torch.cuda.empty_cache()
    gc.collect()

