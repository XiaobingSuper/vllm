"""
Attention computation layer with vLLM-specific attention sink logic,
as described in https://github.com/mit-han-lab/streaming-llm.
"""
from typing import List, Optional, Tuple
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding



class SnapKVCluster:
    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(
        self,
        key_states,
        query_states,
        value_states
    ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.window_size :, :], key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[
                :, :, -self.window_size :, -self.window_size :
            ] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights_sum = attn_weights[
                :, :, -self.window_size :, : -self.window_size
            ].sum(dim=-2)
            if self.pooling == "avgpool":
                attn_cache = F.avg_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            elif self.pooling == "maxpool":
                attn_cache = F.max_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            else:
                raise ValueError("Pooling method not supported")
            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1
            ).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, "window_size"):
            self.config.window_size = 64
        if not hasattr(self.config, "max_capacity_prompt"):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, "kernel_size"):
            self.config.kernel_size = 13
        if not hasattr(self.config, "pooling"):
            self.config.pooling = "avgpool"
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
    )

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
 
_SUPPORTED_ATTN_BACKENDS = (
    _Backend.FLASH_ATTN,
    _Backend.XFORMERS,
    _Backend.FLASHINFER,
)


class SnapKVAttention(nn.Module):
    """Replacement for Attention layer when snapkv are enabled."""

    def __init__(
        self,
        model_context_len: int,
        block_size: int,
        kv_cache_dtype: str,
        attn_backend: _Backend,
        num_kv_heads: int,
        head_dim: int,
        rotary_emb_layer: Optional[RotaryEmbedding],
        attn_layer: Attention,
        chunked_prefill_enabled: bool
    ) -> None:
        super().__init__()
        self.model_context_len = model_context_len
        self.block_size = block_size
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_backend = attn_backend
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb_layer
        self.use_alibi = rotary_emb_layer is None
        self.attn = attn_layer
        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.positions = None

        if attn_backend not in _SUPPORTED_ATTN_BACKENDS:
            raise NotImplementedError(
                'Attention sinks is only supported for '
                'FlashAttention, XFormers, and FlashInfer currently.')

    def save_positions(self, positions: torch.Tensor):
        self.positions = positions

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Replaces the `self.attn(...)` call in model attention modules."""
        
        init_snapkv(self)
        if kv_cache.numel() == 0:
            # ModelRunner.profile_run
            if not self.use_alibi:
                q, k = self.rotary_emb(self.positions, q, k)
            return self.attn(q, k, v, kv_cache, attn_metadata)
     
        if k.shape[-2] == q.shape[-2]: # [SnapKV] add kv_cluster for prefill path
            if self.use_alibi:
                q, k = self.rotary_emb(self.positions, q, k)
            # direct using k, v
            
            output = self.attn(q, k, v, kv_cache, attn_metadata)
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            k, v = self.kv_cluster.update_kv(k, q, v)
            # write kv back to kv_cache
            
            # update positions            
        else:
            if self.use_alibi:
                # update positions
                q, k = self.rotary_emb(self.positions, q, k)
            return self.attn(q, k, v, kv_cache, attn_metadata)