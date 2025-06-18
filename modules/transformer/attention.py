import torch
import torch.nn.functional as F

from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from torch import nn
from torch import Tensor
from torch.nested import Tensor as NestedTensor
from typing import Optional
from typing import Union

torch.backends.cuda.enable_flash_sdp(True)

AttentionInput = Union[Tensor, NestedTensor]

class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency inverse
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embedding to input tensor x"""
    # Split the last dimension into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    # cos and sin shapes: (seq_len, head_dim//2)
    # x1, x2 shapes: (..., seq_len, head_dim//2)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Interleave back
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    return rotated_x


class KVCache(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert len(dim) == 3, "Cache only supports 3d tensors"
        self.register_buffer("k_cache", torch.zeros(*dim, requires_grad=False))
        self.register_buffer("v_cache", torch.zeros(*dim, requires_grad=False))
        self.dim = dim

        self._reset_limits()
        self.is_empty = True

    def _reset_limits(self):
        self.cache_limits = [0 for _ in self.dim]
        self.next_seq_pos = None

    def reset(self):
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)

        self._reset_limits()
        self.is_empty = True

    @property
    def device(self):
        return self.k_cache.device

    @property
    def keys(self):
        B, N, D = self.cache_limits
        return self.k_cache[:B, :N, :D]

    @property
    def values(self):
        B, N, D = self.cache_limits
        return self.v_cache[:B, :N, :D]

    @property
    def seq_lengths(self):
        if self.is_empty:
            return 0
        return self.next_seq_pos

    @torch.no_grad
    def store(self, keys: Tensor, values: Tensor, mask: Tensor) -> None:
        B, N = mask.shape
        self.k_cache[:B, :N, :][mask] = keys.detach()[:, :]
        self.v_cache[:B, :N, :][mask] = values.detach()[:, :]

        self.cache_limits = [B, N, self.dim[-1]]
        self.next_seq_pos = mask.sum(axis=1).unsqueeze(-1)
        self.is_empty = False

    @torch.no_grad
    def append_column(self, keys: Tensor, values: Tensor) -> None:
        B, N, D = self.cache_limits

        row_idx = torch.arange(B, device=self.k_cache.device)
        self.k_cache[:B, :][row_idx, self.next_seq_pos] = keys.detach()[:, :]
        self.v_cache[:B, :][row_idx, self.next_seq_pos] = values.detach()[:, :]

        max_pos_appended = self.next_seq_pos.max()
        if max_pos_appended >= N:
            self.cache_limits[1] = max_pos_appended + 1
        self.next_seq_pos += 1

    @torch.no_grad
    @torch.compiler.disable
    def as_jagged(self):
        keys_jagged = padded_to_jagged_tensor(
            self.keys, lengths=self.seq_lengths.squeeze(), max_len=self.keys.shape[1]
        )
        values_jagged = padded_to_jagged_tensor(
            self.values,
            lengths=self.seq_lengths.squeeze(),
            max_len=self.values.shape[1],
        )
        return keys_jagged, values_jagged

    @torch.no_grad
    def apply(self, fn) -> None:
        B, N, D = self.cache_limits
        k_transformed, v_transformed = fn(self.k_cache[:B, :N, :D]), fn(
            self.v_cache[:B, :N, :D]
        )
        next_seq_pos_transformed = fn(self.next_seq_pos)
        B, N, D = k_transformed.shape

        self.reset()
        self.k_cache[:B, :N, :D] = k_transformed
        self.v_cache[:B, :N, :D] = v_transformed
        self.next_seq_pos = next_seq_pos_transformed
        self.cache_limits = [B, N, D]
        self.is_empty = False


class Attend(nn.Module):
    def __init__(self, d_out, num_heads, head_dim, dropout, use_rope=False):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out
        self.dropout = dropout
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPEEmbedding(self.head_dim)

    def jagged_forward(
        self, qu: NestedTensor, ke: NestedTensor, va: NestedTensor, is_causal: bool
    ) -> NestedTensor:
        queries = qu.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        keys = ke.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        values = va.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        if self.use_rope:
            # apply RoPE to queries and keys
            seq_len = queries.shape[-2]
            cos, sin = self.rope(queries, seq_len)
            # reshape cos/sin for broadcasting: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            # apply rope
            queries = apply_rotary_pos_emb(queries, cos, sin)
            keys = apply_rotary_pos_emb(keys, cos, sin)

        dropout_p = 0.0 if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=dropout_p, is_causal=is_causal
        )

        context_vec = context_vec.transpose(1, 2).flatten(-2)
        return context_vec

    def forward(self, qkv: Tensor, is_causal: bool = False) -> Tensor:
        batch_size, num_tokens, embed_dim = qkv.shape
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        if self.use_rope:
            # apply RoPE to queries and keys
            cos, sin = self.rope(queries, num_tokens)
            # reshape cos/sin for broadcasting: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            # apply rope
            queries = apply_rotary_pos_emb(queries, cos, sin)
            keys = apply_rotary_pos_emb(keys, cos, sin)

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=is_causal,
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        cross_attn=False,
        dropout=0.0,
        qkv_bias=False,
        enable_kv_cache=False,
        rope=False,
    ) -> None:
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"
        assert not enable_kv_cache, "KV Cache currently not supported"

        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.enable_kv_cache = enable_kv_cache

        if self.cross_attn:
            self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.kv = nn.Linear(d_in, 2 * d_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        self.proj = nn.Linear(d_out, d_out, bias=False)

        self.attend = Attend(self.d_out, self.num_heads, self.head_dim, dropout=False, use_rope=rope)

        self._kv_cache = (
            KVCache((2560, 80, 384)) if enable_kv_cache else None
        )  # (640, 800, 64) TODO: Revisit KV Cache

    @property
    def kv_cache(self) -> KVCache:
        return self._kv_cache

    def forward(
        self,
        x: AttentionInput,
        x_kv: Optional[AttentionInput] = None,
        padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = True,
        jagged: bool = False,
        use_cache: bool = False,
    ) -> AttentionInput:
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        assert (
            not self.cross_attn or x_kv is not None
        ), "Found null x_kv in cross attn. layer"

        if self.cross_attn:
            queries = self.q(x)
            keys, values = self.kv(x_kv).chunk(2, dim=-1)
        else:
            queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        if (
            not self.training
            and use_cache
            and self.enable_kv_cache
            and self.kv_cache.is_empty
        ):
            assert padding_mask is not None
            B, N = padding_mask.shape

            self.kv_cache.store(
                keys=jagged_to_flattened_tensor(keys),
                values=jagged_to_flattened_tensor(values),
                mask=padding_mask,
            )
            context_vec = self.attend.jagged_forward(
                queries, keys, values, is_causal=is_causal
            )

        elif (
            not self.training
            and use_cache
            and self.enable_kv_cache
            and not self.kv_cache.is_empty
        ):
            assert padding_mask is not None
            B, N = padding_mask.shape

            keys, values = jagged_to_flattened_tensor(keys), jagged_to_flattened_tensor(
                values
            )

            self.kv_cache.append_column(keys=keys, values=values)
            keys, values = self.kv_cache.as_jagged()

            context_vec = self.attend.jagged_forward(
                queries, keys, values, is_causal=False
            )

        elif jagged:
            context_vec = self.attend.jagged_forward(
                queries, keys, values, is_causal=is_causal
            )

        if not jagged:
            raise Exception("Unjagged attention currently not supported.")
            # context_vec = self.attend(qkv, is_causal=is_causal)

        context_vec = self.proj(context_vec)
        return context_vec
