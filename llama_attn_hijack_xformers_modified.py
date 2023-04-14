import math
import sys
import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama

from typing import Optional
from typing import Tuple

import xformers.ops

# import modules.shared as shared

# if shared.args.xformers:
#     try:
#     except Exception:
#         print("Please install xformers before trying to use it", file=sys.stderr)

def hijack_llama_attention():
    # if shared.args.xformers:
    transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward
    print("Replaced attention with xformers_attention")

def inplace_concat(past_state, new_state):
    _, _, past_seq_len, _ = past_state.size()
    _, _, new_seq_len, _ = new_state.size()
    total_seq_len = past_seq_len + new_seq_len

    # Expand the past_state tensor to accommodate the new states
    expanded_state = torch.zeros(past_state.size(0), past_state.size(1), total_seq_len, past_state.size(3), dtype=past_state.dtype, device=past_state.device)
    expanded_state.narrow(2, 0, past_seq_len).copy_(past_state)

    # Update the target_slice with new_state values
    expanded_state[:, :, past_seq_len:total_seq_len, :].copy_(new_state)

    return expanded_state


def inplace_copy_and_view(target, source):
    # Ensure the shapes of the target and source tensors are compatible
    assert target.size(0) == source.size(0) and target.size(1) == source.size(1) and target.size(3) == source.size(
        3), "Incompatible tensor shapes"

    # Ensure the target tensor has enough space along dimension 2
    assert target.size(2) >= source.size(2), "Target tensor has insufficient space along dimension 2"

    # Create a view of the target tensor along dimension 2 and copy the source tensor into it
    target.narrow(2, 0, source.size(2)).copy_(source)

    # Create a view of the target tensor that matches the shape of the source tensor
    target_view = target.narrow(2, 0, source.size(2))

    return target_view

MAX_KV_LENGTH = 8192

def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_value is None:
        p_key_states = torch.zeros(bsz, self.num_heads, MAX_KV_LENGTH, self.head_dim, dtype=key_states.dtype, device=key_states.device)
        p_value_states = torch.zeros(bsz, self.num_heads, MAX_KV_LENGTH, self.head_dim, dtype=key_states.dtype, device=key_states.device)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        p_key_states = inplace_copy_and_view(p_key_states, key_states)
        p_value_states = inplace_copy_and_view(p_value_states, value_states)

        past_key_value = (p_key_states, p_value_states) if use_cache else None

    else:
        kv_seq_len = key_states.shape[-2]
        kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        o_shape = (past_key_value[0].shape[0], past_key_value[0].shape[1], MAX_KV_LENGTH, past_key_value[0].shape[3])
        o_offset = past_key_value[0].storage_offset()
        o_stride = past_key_value[0].stride()

        rec_key_states = torch.empty(o_shape, dtype=past_key_value[0].dtype, device=past_key_value[0].device)
        rec_key_states.set_(past_key_value[0].storage(), storage_offset = o_offset, size = o_shape, stride = o_stride)

        rec_value_states = torch.empty(o_shape, dtype=past_key_value[1].dtype, device=past_key_value[1].device)
        rec_value_states.set_(past_key_value[1].storage(), storage_offset = o_offset, size = o_shape, stride = o_stride)

        new_key_view = rec_key_states.narrow(2, past_key_value[0].shape[2], key_states.shape[2])
        new_value_view = rec_value_states.narrow(2, past_key_value[1].shape[2], value_states.shape[2])
        new_key_view.copy_(key_states)
        new_value_view.copy_(value_states)

        p_key_states = rec_key_states.narrow(2, 0, kv_seq_len)
        p_value_states = rec_value_states.narrow(2, 0, kv_seq_len)

        past_key_value = (p_key_states, p_value_states) if use_cache else None

    # else:
    #     # reuse k, v, self_attention
    #     # key_states2 = torch.cat([past_key_value[0], key_states], dim=2)
    #     # value_states2 = torch.cat([past_key_value[1], value_states], dim=2)
    #     key_states = inplace_concat(past_key_value[0], key_states)
    #     value_states = inplace_concat(past_key_value[1], value_states)

    #We only apply xformers optimizations if we don't need to output the whole attention matrix
    if not output_attentions:
        dtype = query_states.dtype

        query_states = query_states.transpose(1, 2)
        key_states = p_key_states.transpose(1, 2)
        value_states = p_value_states.transpose(1, 2)

        #This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        #We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=None)
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=xformers.ops.LowerTriangularMask())
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value