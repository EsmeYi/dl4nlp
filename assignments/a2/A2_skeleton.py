import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        H, I = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj   = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: down_proj( gate_proj(x) * SiLU(up_proj(x)) )
        return self.down_proj(self.gate_proj(hidden_states) * self.act(self.up_proj(hidden_states)))

class A2RMSNorm(nn.Module):
    """RMS layer normalization (manual implementation for PyTorch < 2.4)."""
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

def make_norm(config):
    return A2RMSNorm(config)


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head  = H // self.n_heads       # dimension per head

        # Four projection matrices, all H→H, no bias (OLMo 2 style)
        self.W_q = nn.Linear(H, H, bias=False)
        self.W_k = nn.Linear(H, H, bias=False)
        self.W_v = nn.Linear(H, H, bias=False)
        self.W_o = nn.Linear(H, H, bias=False)

        # OLMo 2 applies RMSNorm after the Q and K projections
        self.q_norm = make_norm(config)
        self.k_norm = make_norm(config)

    def forward(self, hidden_states, rope_rotations):
        B, N, H = hidden_states.shape
        n_h, d_h = self.n_heads, self.d_head

        # Step 1: project to Q, K, V and normalise Q and K
        q = self.q_norm(self.W_q(hidden_states))   # (B, N, H)
        k = self.k_norm(self.W_k(hidden_states))   # (B, N, H)
        v = self.W_v(hidden_states)                # (B, N, H)

        # Step 2: split into heads → (B, n_h, N, d_h)
        q = q.view(B, N, n_h, d_h).transpose(1, 2)
        k = k.view(B, N, n_h, d_h).transpose(1, 2)
        v = v.view(B, N, n_h, d_h).transpose(1, 2)

        # Step 3: apply RoPE positional encoding to Q and K
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        # Step 4: causal scaled dot-product attention (PyTorch handles mask + softmax)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Step 5: merge heads back → (B, N, H) then output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, N, H)
        return self.W_o(attn_out)


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.attn_norm = make_norm(config)   # norm before attention
        self.attention = A2Attention(config)
        self.mlp_norm  = make_norm(config)   # norm before MLP
        self.mlp       = A2MLP(config)

    def forward(self, hidden_states, rope_rotations):
        # Attention sub-layer with residual connection
        residual    = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, rope_rotations)
        hidden_states = hidden_states + residual          # residual

        # MLP sub-layer with residual connection
        residual    = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual          # residual

        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb  = A2RotaryEmbedding(config)
        self.embedding   = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers      = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm  = make_norm(config)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_func   = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()

    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids)

        # Embedding: (B, N) → (B, N, H)
        h = self.embedding(input_ids)

        # Pass through each Transformer decoder layer
        for layer in self.layers:
            h = layer(h, rope_rotations)

        # Final normalisation + unembedding: (B, N, H) → (B, N, V)
        h      = self.final_norm(h)
        logits = self.unembedding(h)

        # Compute loss if labels provided (same shift trick as A1)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(logits=logits, loss=loss)


###
### Task 3.2: Text generation
###

def generate(model, tokenizer, prompt, max_length=100, temperature=1.0, topk=None, device='cpu'):
    """Generate text autoregressively from a prompt.

    Args:
        model:       The language model.
        tokenizer:   The tokenizer (A1Tokenizer).
        prompt:      The input string to continue from.
        max_length:  Maximum number of new tokens to generate.
        temperature: Scales logits before sampling. Lower = more deterministic.
        topk:        If set, only sample from the top-k highest-probability tokens.
        device:      'cpu' or 'cuda'.
    """
    import torch
    from torch.distributions import Categorical

    model.eval()
    # Encode prompt (no padding needed for single sequence)
    input_ids = tokenizer([prompt], return_tensors='pt', padding=False, truncation=True)['input_ids']
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_length):
            out = model(input_ids)

            # Take logits at the last real token (position -2 to avoid EOS if present,
            # but here we just take the very last position since we control the sequence)
            next_logits = out.logits[0, -1, :]   # shape: (V,)

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k filtering: set all logits outside top-k to -inf
            if topk is not None:
                topk_vals = next_logits.topk(topk).values[-1]   # k-th largest value
                next_logits[next_logits < topk_vals] = float('-inf')

            # Sample from the distribution
            next_token = Categorical(logits=next_logits).sample().unsqueeze(0).unsqueeze(0)  # (1,1)

            # Stop if EOS generated
            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode back to string (skip BOS token at position 0)
    tokens = [tokenizer.int_to_str.get(i.item(), '<UNK>') for i in input_ids[0][1:]]
    return ' '.join(tokens)


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
