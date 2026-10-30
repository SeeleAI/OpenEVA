from typing import Optional, Sequence, Union, overload
import torch
from .. import VarLenTensor
from .. import SparseTensor
from .. import config


__all__ = [
    'sparse_scaled_dot_product_attention',
]


def _cached_flash_cu_seqlens(
    owner: Optional[VarLenTensor],
    sequence_lengths: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Build FlashAttention offsets once per persistent sparse layout.

    Native Shape/Material self-attention keeps the same sparse coordinates for
    every transformer slot and for activation-checkpoint recomputation.  The
    old path rebuilt a pageable CPU ``int32`` tensor and copied it to CUDA on
    every call, which synchronized the preceding CUDA queue even though the
    sequence layout had not changed.  ``SparseTensor.replace`` and
    ``VarLenTensor.replace`` already preserve their layout caches, so keep the
    tiny offset tensor with that owner instead.
    """

    lengths = tuple(int(length) for length in sequence_lengths)
    device = torch.device(device)
    cache_key = (
        "flash_attn_cu_seqlens_int32",
        device.type,
        device.index,
        lengths,
    )
    cached = None
    if isinstance(owner, SparseTensor):
        cached = owner.get_spatial_cache(cache_key)
    elif isinstance(owner, VarLenTensor):
        cached = owner._cache.get(cache_key)
    if cached is not None:
        return cached

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cached = torch.tensor(offsets, dtype=torch.int32, device=device)
    if isinstance(owner, SparseTensor):
        owner.register_spatial_cache(cache_key, cached)
    elif isinstance(owner, VarLenTensor):
        owner._cache[cache_key] = cached
    return cached


def _build_full_block_mask(q_seqlen, kv_seqlen, device):
    if 'create_block_mask' not in globals():
        from torch.nn.attention.flex_attention import create_block_mask

    max_q_seqlen = max(q_seqlen)
    max_kv_seqlen = max(kv_seqlen)
    q_lens = torch.tensor(q_seqlen, dtype=torch.long, device=device)
    kv_lens = torch.tensor(kv_seqlen, dtype=torch.long, device=device)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx < q_lens[b]) & (kv_idx < kv_lens[b])

    return create_block_mask(
        mask_mod,
        B=len(q_seqlen),
        H=None,
        Q_LEN=max_q_seqlen,
        KV_LEN=max_kv_seqlen,
        device=device,
        BLOCK_SIZE=128,
        _compile=False,
    )


def _dense_from_layout(varlen: Union[VarLenTensor, SparseTensor], max_length: int):
    batch_size = len(varlen.layout)
    feature_shape = tuple(varlen.feats.shape[1:])
    dense = torch.zeros(batch_size, max_length, *feature_shape, dtype=varlen.dtype, device=varlen.device)
    for batch_index, layout in enumerate(varlen.layout):
        seq = varlen.feats[layout]
        dense[batch_index, : seq.shape[0]] = seq
    return dense


def _varlen_from_dense(template: Union[VarLenTensor, SparseTensor], dense: torch.Tensor, q_seqlen):
    feats = torch.cat([dense[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(q_seqlen)], dim=0)
    return template.replace(feats)


def _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask):
    if 'flex_attention_hop' not in globals():
        from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
    q_dense = q_dense.permute(0, 2, 1, 3)   # [N, H, Lq, C]
    k_dense = k_dense.permute(0, 2, 1, 3)   # [N, H, Lk, C]
    v_dense = v_dense.permute(0, 2, 1, 3)   # [N, H, Lk, C]
    def score_mod(score, b, h, q_idx, kv_idx):
        del b, h, q_idx, kv_idx
        return score
    out, _lse = flex_attention_hop(
        q_dense,
        k_dense,
        v_dense,
        score_mod,
        block_mask.as_tuple(),
        q_dense.shape[-1] ** -0.5,
        {},
    )
    return out.permute(0, 2, 1, 3)   # [N, Lq, H, C]


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (VarLenTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, kv: Union[VarLenTensor, torch.Tensor]) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (VarLenTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (VarLenTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: VarLenTensor, v: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: torch.Tensor, v: torch.Tensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: VarLenTensor, v: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_owner = qkv
        kv_owner = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_owner = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            q_owner = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_owner = kv
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            kv_owner = None
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) is type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_owner = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            q_owner = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_owner = k
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            kv_owner = None
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        cu_seqlens_q = _cached_flash_cu_seqlens(q_owner, q_seqlen, device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = _cached_flash_cu_seqlens(
                kv_owner,
                kv_seqlen,
                device,
            )
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif config.ATTN == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = _cached_flash_cu_seqlens(q_owner, q_seqlen, device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = _cached_flash_cu_seqlens(
                kv_owner,
                kv_seqlen,
                device,
            )
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = _cached_flash_cu_seqlens(
                kv_owner,
                kv_seqlen,
                device,
            )
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    elif config.ATTN == 'flex_attention':
        max_q_seqlen = max(q_seqlen)
        max_kv_seqlen = max(kv_seqlen)
        block_mask = _build_full_block_mask(q_seqlen, kv_seqlen, device)

        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            q_dense = _dense_from_layout(VarLenTensor(q, VarLenTensor.layout_from_seqlen(q_seqlen)), max_q_seqlen)
            k_dense = _dense_from_layout(VarLenTensor(k, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
            v_dense = _dense_from_layout(VarLenTensor(v, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
            out_dense = _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask)
            out = _varlen_from_dense(s, out_dense, q_seqlen)
            return out

        if num_all_args == 2:
            if isinstance(args[0] if len(args) > 0 else kwargs['q'], VarLenTensor):
                k, v = kv.unbind(dim=1)
                q_dense = _dense_from_layout(s, max_q_seqlen)
                k_dense = _dense_from_layout(VarLenTensor(k, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
                v_dense = _dense_from_layout(VarLenTensor(v, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
                out_dense = _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask)
                out = _varlen_from_dense(s, out_dense, q_seqlen)
                return out

            k, v = kv.unbind(dim=1)
            q_dense = q.reshape(len(q_seqlen), max_q_seqlen, q.shape[1], q.shape[2])
            k_dense = _dense_from_layout(VarLenTensor(k, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
            v_dense = _dense_from_layout(VarLenTensor(v, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
            out = _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask)
            return out

        if isinstance(args[0] if len(args) > 0 else kwargs['q'], VarLenTensor):
            q_dense = _dense_from_layout(s, max_q_seqlen)
            if isinstance(args[1] if len(args) > 1 else kwargs['k'], VarLenTensor):
                k_dense = _dense_from_layout(VarLenTensor(k, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
                v_dense = _dense_from_layout(VarLenTensor(v, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
            else:
                k_dense = k.reshape(len(kv_seqlen), max_kv_seqlen, k.shape[1], k.shape[2])
                v_dense = v.reshape(len(kv_seqlen), max_kv_seqlen, v.shape[1], v.shape[2])
            out_dense = _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask)
            out = _varlen_from_dense(s, out_dense, q_seqlen)
            return out

        q_dense = q.reshape(len(q_seqlen), max_q_seqlen, q.shape[1], q.shape[2])
        k_dense = _dense_from_layout(VarLenTensor(k, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
        v_dense = _dense_from_layout(VarLenTensor(v, VarLenTensor.layout_from_seqlen(kv_seqlen)), max_kv_seqlen)
        return _flex_varlen_attention(q_dense, k_dense, v_dense, block_mask)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
