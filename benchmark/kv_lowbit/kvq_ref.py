"""Reference (pure PyTorch) KV cache quantizers for the low-bit KV project.

Each quantizer maps a K or V tensor whose last dim is head_dim to its
quantize-then-dequantize value in the input dtype. The per-token-head schemes
use one scale per (token, head) vector, so quantizing a slice of tokens gives
the same result as quantizing the whole tensor at once.

The INT8 / INT4 schemes follow vLLM's ``int8_per_token_head`` and
``int4_per_token_head`` at commit 9bc1c8b:
``vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`` (INT8) and
``vllm/v1/attention/ops/int4_per_token_head.py`` (INT4, RHT).

The RHT variant returns K/V rotated back to the original basis, so the result
can be fed to an unmodified attention. vLLM instead keeps the cache rotated and
rotates Q and the output; the two agree up to one extra BF16 rounding of the
dequantized values.
"""

import torch

# vLLM's _get_rht_signs seed for round 0.
_RHT_SEED = 0x9E3779B9

_HADAMARD_CACHE: dict = {}
_SIGNS_CACHE: dict = {}


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, x - 0.5).trunc()


def _largest_power_of_2_divisor(n: int) -> int:
    return n & -n


def _hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix (H @ H = d * I), float32."""
    key = (d, str(device))
    if key not in _HADAMARD_CACHE:
        h = torch.ones(1, 1)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HADAMARD_CACHE[key] = h.to(device)
    return _HADAMARD_CACHE[key]


def _rht_signs(d: int, device: torch.device) -> torch.Tensor:
    """The same deterministic +-1 vector vLLM draws for round 0."""
    key = (d, str(device))
    if key not in _SIGNS_CACHE:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_RHT_SEED)
        signs = 2.0 * torch.bernoulli(torch.full((d,), 0.5), generator=gen) - 1.0
        _SIGNS_CACHE[key] = signs.to(device)
    return _SIGNS_CACHE[key]


def rht(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """vLLM's single RHT in float32: forward H(D x), inverse D H(x) / b.

    Non-power-of-2 head sizes are rotated block-diagonally with b, the largest
    power-of-2 divisor of head_dim, as vLLM does.
    """
    d = x.shape[-1]
    b = _largest_power_of_2_divisor(d)
    xf = x.float().reshape(*x.shape[:-1], d // b, b)
    h = _hadamard(b, x.device)
    signs = _rht_signs(b, x.device)
    if inverse:
        y = (xf @ h) * signs / b
    else:
        y = (xf * signs) @ h
    return y.reshape(x.shape)


def fp8_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """FP8 KV without calibrated scales (scale 1.0), the engines' default."""
    return x.to(dtype).to(x.dtype)


def int8_per_token_head(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    scale = (xf.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-6)
    q = _round_half_away(xf * (1.0 / scale)).clamp(-128.0, 127.0)
    return (q * scale).to(x.dtype)


def _strip_low_mantissa(scale: torch.Tensor) -> torch.Tensor:
    # vLLM hides the 4-bit zero-point in the scale's low mantissa bits and
    # dequantizes with those bits cleared.
    return (scale.contiguous().view(torch.int32) & -16).view(torch.float32)


def int4_per_token_head(x: torch.Tensor, use_rht: bool = True) -> torch.Tensor:
    if use_rht:
        # vLLM rotates in float32 and casts back before quantizing.
        xf = rht(x).to(x.dtype).float()
    else:
        xf = x.float()
    xmin = xf.amin(-1, keepdim=True)
    xmax = xf.amax(-1, keepdim=True)
    scale = ((xmax - xmin) / 15.0).clamp_min(1e-6)
    zp = _round_half_away(-xmin / scale).clamp(0.0, 15.0)
    q = _round_half_away(xf * (1.0 / scale) + zp).clamp(0.0, 15.0)
    deq = (q - zp) * _strip_low_mantissa(scale)
    if use_rht:
        deq = rht(deq, inverse=True)
    return deq.to(x.dtype)


def fp4_mx_block16(x: torch.Tensor) -> torch.Tensor:
    """SGLang's own fp4_mx_block16 recipe (E2M1, one UE8M0 scale per 16)."""
    from sglang.srt.layers.quantization.kvfp4_tensor import (
        FP4MXBlock16KVQuantizeUtil,
    )

    d = x.shape[-1]
    flat = x.reshape(-1, 1, d).contiguous()
    packed, scales = FP4MXBlock16KVQuantizeUtil.batched_quantize(flat)
    deq = FP4MXBlock16KVQuantizeUtil.batched_dequantize(packed, scales, dtype=x.dtype)
    return deq.reshape(x.shape)


SCHEMES = {
    "fp8_e5m2": lambda x: fp8_cast(x, torch.float8_e5m2),
    "fp8_e4m3": lambda x: fp8_cast(x, torch.float8_e4m3fn),
    "int8_pth": int8_per_token_head,
    "int4_pth_rht": lambda x: int4_per_token_head(x, use_rht=True),
    "int4_pth_norht": lambda x: int4_per_token_head(x, use_rht=False),
    "fp4_mx_block16": fp4_mx_block16,
}


def channel_outlier_ratio(k: torch.Tensor) -> torch.Tensor:
    """Per-head max/median of the per-channel absmax over tokens.

    ``k`` is ``[..., tokens, head_dim]``; returns ``[...]``. A large ratio
    means a few channels carry most of the magnitude, which hurts per-token
    quantization.
    """
    absmax = k.float().abs().amax(dim=-2)
    return absmax.amax(-1) / absmax.median(-1).values.clamp_min(1e-12)
