"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    L_valid, D: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    valid_rows = offsets < L_valid

    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets).to(tl.float32)

    # mean over real rows only (zero padding would bias the tail group), accumulated in fp32,
    # then rounded once so the qm subtracted here is exactly the qm used for delta_s
    num_valid = tl.minimum(L_valid - group_start, GROUP_SIZE)
    qm_group = (tl.sum(q_group, axis=0) / num_valid).to(qm_out_ptr.dtype.element_ty)

    q_group = tl.where(valid_rows[:, None], q_group - qm_group.to(tl.float32), 0.0)
    tl.store(q_out_ptr + q_offsets, q_group.to(q_out_ptr.dtype.element_ty))

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor, valid_len: int):
    """Subtract each 128-row group's mean from a 128-padded ``q``; only its first ``valid_len`` rows are real."""
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)

    group_mean_kernel[grid](
        q, q_out, qm,
        valid_len, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):

    def pad_128(x, shift=None):
        """Zero-pad the sequence dim to a multiple of 128, subtracting ``shift`` from the real rows if given."""
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if shift is None:
            if pad_len == 0:
                return x.contiguous()
            return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
        # x - shift goes straight into a fresh padded buffer: one copy, and the caller's tensor is never modified
        out = x.new_empty(x.size(0), x.size(1), L + pad_len, x.size(3))
        out[:, :, L:].zero_()
        torch.sub(x, shift, out=out[:, :, :L])
        return out

    q_len = q.size(2)
    k = pad_128(k, k.mean(dim=-2, keepdim=True))
    v = pad_128(v)
    if per_block_mean:
        q, qm = triton_group_mean(pad_128(q), q_len)
    else:
        # mean over the real rows only, so zero padding does not bias it
        qm = q.mean(dim=-2, keepdim=True)
        q = pad_128(q, qm)
    # the kernel reads delta_s as fp32, so autocast must not lower this GEMM to 16 bits
    with torch.autocast(device_type=q.device.type, enabled=False):
        try:
            # 16-bit GEMM with fp32 accumulation and output: products are exact and K is not upcast
            delta_s = torch.bmm(qm.flatten(0, 1), k.flatten(0, 1).transpose(1, 2), out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            # torch < 2.8 has no out_dtype, and fp16 under allow_fp16_accumulation rejects an fp32 output;
            # fp16/bf16 values are exact in tf32, so the upcast GEMM loses nothing
            delta_s = torch.bmm(qm.flatten(0, 1).float(), k.flatten(0, 1).float().transpose(1, 2))
    return q, k, v, delta_s.unflatten(0, qm.shape[:2])

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    KL,
    is_causal,
    per_block_mean,
    is_bf16
    )[0][:, :, :QL, :].contiguous()
    return o_fp4