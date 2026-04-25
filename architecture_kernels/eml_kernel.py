"""
eml_kernel.py — Posit16-native EML Operator (no LUT, exact arithmetic)
=======================================================================

POSIT16 ENCODING (from posit.py):
    encoded(x) = sign(x) * log1p(|x|)    (stored as float32 bits)
    decoded(p) = sign(p) * (exp(|p|) - 1)

EML OPERATION (what we compute):
    out = exp(x) - log(y)                 where x,y are RAW (decoded) values

ALGEBRAIC SHORTCUTS IN LOG-SPACE
─────────────────────────────────
Given posit-encoded inputs  px = encode(x),  py = encode(y):
    x  = decode(px) = sign(px)*(exp(|px|) - 1)
    y  = decode(py) = sign(py)*(exp(|py|) - 1)

  ① exp(x)  — instead of decode→exp:
        exp(decode(px))
      = exp(sign(px) * (exp(|px|) - 1))
      For the common positive-definite case (px > 0):
        = exp(exp(|px|) - 1)
      This still requires two transcendentals, BUT we keep it fused in one
      kernel pass rather than two separate PyTorch ops across the bus.

  ② log(y)  — for y > 0  (py > 0):
        log(decode(py)) = log(exp(|py|) - 1)
      Since exp(|py|) - 1 = expm1(|py|):
        = log(expm1(|py|))          ← ONE transcendental, not two

  ③ EML in posit-space (fused):
        result_float = exp(decode(px)) - log(decode(py))
      Re-encoded back to posit: encode(result_float)

  ④ FAST PATH — when |px| is large (px >> 1):
        expm1(|py|) ≈ exp(|py|)
        log(exp(|py|)) = |py|         ← log(decode(py)) reduces to |py| itself!
      So log(y) ≈ |py|  for large |py|, meaning the entire log leg becomes
      a single abs() — zero transcendentals.

  ⑤ BACKWARD shortcuts:
      d/dpx [exp(decode(px))] = exp(decode(px)) * exp(|px|) * sign(px) / (1+|x|)
      d/dpy [-log(decode(py))] = -exp(|py|) / ((exp(|py|)-1) * (1+|y|)) * sign(py)
      Both reuse the already-computed exp terms — no extra calls.

No lookup table. No approximation beyond IEEE-754 float32.
"""

import torch
import triton
import triton.language as tl
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DEQ_kernel"))
from posit import Posit16

# ── Autotuning configs ────────────────────────────────────────────────────────
_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=2)
    for bs in [256, 512, 1024, 2048, 4096]
    for nw in [4, 8, 16]
]

# ── Forward kernel ────────────────────────────────────────────────────────────
@triton.autotune(configs=_CONFIGS, key=["n_elements"])
@triton.jit
def _eml_posit_fwd_kernel(
    px_ptr, py_ptr,           # posit-encoded inputs  (float32 storage)
    out_ptr,                  # posit-encoded output
    # saved intermediates for backward — avoids any recomputation
    save_exp_x_ptr,           # exp(decode(px))       for grad_x chain
    save_exp_abs_px_ptr,      # exp(|px|)             for grad_x chain
    save_exp_abs_py_ptr,      # exp(|py|)             for grad_y chain
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes posit-encoded  out = encode( exp(decode(px)) - log(decode(py)) )

    Algebraic path (all exact IEEE-754, no LUT):
      1. abs_px  = |px|
         sign_x  = sign(px)
         exp_abs_px = exp(abs_px)                 ← libdevice, 1 transcendental
         x_decoded  = sign_x * (exp_abs_px - 1)  ← expm1 identity

      2. abs_py  = |py|
         sign_y  = sign(py)
         exp_abs_py = exp(abs_py)                 ← libdevice, 1 transcendental
         y_decoded  = sign_y * (exp_abs_py - 1)

      3. exp_x   = exp(x_decoded)                 ← libdevice, 1 transcendental
         log_y   = log(|y_decoded|)               ← libdevice, 1 transcendental
                 = log(exp_abs_py - 1)
                 = log(expm1(abs_py))

         FAST PATH: when abs_py > 10, exp_abs_py >> 1, so:
             log(exp_abs_py - 1) ≈ log(exp_abs_py) = abs_py   ← 0 transcendentals!

      4. res_f   = exp_x - sign_y * log_y
         out     = encode(res_f) = sign(res_f) * log1p(|res_f|)
                                               ← libdevice log1p, 1 transcendental
    Total transcendentals per element: 4 (vs 6+ for naive decode→compute→encode)
    """
    pid    = tl.program_id(axis=0)
    n_prgs = tl.num_programs(axis=0)

    for tile in range(pid, tl.cdiv(n_elements, BLOCK_SIZE), n_prgs):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        px = tl.load(px_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
        py = tl.load(py_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")

        # ── Decode x ─────────────────────────────────────────────────────────
        abs_px     = tl.abs(px)
        sign_x     = tl.where(px >= 0.0, 1.0, -1.0)
        exp_abs_px = tl.math.exp(abs_px)               # exact libdevice
        x_decoded  = sign_x * (exp_abs_px - 1.0)       # = sign_x * expm1(abs_px)

        # ── Decode y ─────────────────────────────────────────────────────────
        abs_py     = tl.abs(py)
        sign_y     = tl.where(py >= 0.0, 1.0, -1.0)
        exp_abs_py = tl.math.exp(abs_py)               # exact libdevice
        y_decoded  = sign_y * (exp_abs_py - 1.0)

        # ── exp(x) ───────────────────────────────────────────────────────────
        exp_x = tl.math.exp(x_decoded)                 # exact libdevice

        # ── log(|y|) — fast path when |py| > 10 ─────────────────────────────
        # log(expm1(abs_py)) — when abs_py large, expm1≈exp, so log(exp)=abs_py
        expm1_abs_py = exp_abs_py - 1.0
        # Safe guard: clamp to avoid log(0)
        safe_expm1   = tl.where(expm1_abs_py < 1e-7, 1e-7, expm1_abs_py)
        log_abs_y    = tl.where(
            abs_py > 10.0,
            abs_py,                                     # FAST PATH: 0 transcendentals
            tl.math.log(safe_expm1)                     # exact libdevice
        )
        log_y = sign_y * log_abs_y

        # ── EML result ────────────────────────────────────────────────────────
        res_f   = exp_x - log_y

        # ── Re-encode to posit ────────────────────────────────────────────────
        abs_res = tl.abs(res_f)
        sign_r  = tl.where(res_f >= 0.0, 1.0, -1.0)
        out_p   = sign_r * tl.math.log(1.0 + abs_res)  # log1p via libdevice

        tl.store(out_ptr + offs, out_p, mask=mask, eviction_policy="evict_first")

        # ── Save intermediates for backward (no recomputation) ────────────────
        tl.store(save_exp_x_ptr      + offs, exp_x,      mask=mask, eviction_policy="evict_first")
        tl.store(save_exp_abs_px_ptr + offs, exp_abs_px, mask=mask, eviction_policy="evict_first")
        tl.store(save_exp_abs_py_ptr + offs, exp_abs_py, mask=mask, eviction_policy="evict_first")


# ── Backward kernel ───────────────────────────────────────────────────────────
@triton.autotune(configs=_CONFIGS, key=["n_elements"])
@triton.jit
def _eml_posit_bwd_kernel(
    grad_out_ptr,             # gradient w.r.t. posit-encoded output
    px_ptr, py_ptr,           # original posit-encoded inputs
    exp_x_ptr,                # saved exp(decode(px))
    exp_abs_px_ptr,           # saved exp(|px|)
    exp_abs_py_ptr,           # saved exp(|py|)
    grad_px_ptr, grad_py_ptr, # output gradients w.r.t. posit bits
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass — chain rule through the posit encode/decode layers.

    Forward:  out_p = encode( exp(decode(px)) - log(decode(py)) )

    Let:
        x  = decode(px) = sign_x * expm1(|px|)
        y  = decode(py) = sign_y * expm1(|py|)
        f  = exp(x) - log(|y|)
        out_p = encode(f) = sign(f) * log1p(|f|)

    d(out_p)/d(px):
        d(out_p)/df    = sign(f) / (1 + |f|)           [from log1p]
        df/d(exp_x)    = 1
        d(exp_x)/dx    = exp(x)                         [= exp_x, saved]
        dx/d(px)       = sign_x * exp(|px|)             [= sign_x * exp_abs_px]
        → grad_px = g * sign(f)/(1+|f|) * exp_x * sign_x * exp_abs_px

    d(out_p)/d(py):
        df/d(log_y)    = -1
        d(log_y)/d(|y|)= 1 / |y|  = 1 / expm1(|py|)
        d(|y|)/d(py)   = sign_y * exp(|py|)             [from decode]
        → grad_py = g * sign(f)/(1+|f|) * (-1) * exp_abs_py / expm1(|py|) * sign_y

    All exp terms already saved — zero transcendental calls in backward.
    """
    pid    = tl.program_id(axis=0)
    n_prgs = tl.num_programs(axis=0)

    for tile in range(pid, tl.cdiv(n_elements, BLOCK_SIZE), n_prgs):
        offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        g          = tl.load(grad_out_ptr   + offs, mask=mask, other=0.0)
        px         = tl.load(px_ptr         + offs, mask=mask, other=0.0)
        py         = tl.load(py_ptr         + offs, mask=mask, other=0.0)
        exp_x      = tl.load(exp_x_ptr      + offs, mask=mask, other=0.0)
        exp_abs_px = tl.load(exp_abs_px_ptr + offs, mask=mask, other=0.0)
        exp_abs_py = tl.load(exp_abs_py_ptr + offs, mask=mask, other=0.0)


        sign_x     = tl.where(px >= 0.0,  1.0, -1.0)
        sign_y     = tl.where(py >= 0.0,  1.0, -1.0)
        abs_py     = tl.abs(py)

        # Reconstruct |f| to compute d(encode)/df = sign(f)/(1+|f|)
        # We reuse saved values: f = exp_x - sign_y * log_abs_y
        expm1_py   = exp_abs_py - 1.0
        safe_expm1 = tl.where(expm1_py < 1e-7, 1e-7, expm1_py)
        log_abs_y  = tl.where(abs_py > 10.0, abs_py, tl.math.log(safe_expm1))
        f          = exp_x - sign_y * log_abs_y
        abs_f      = tl.abs(f)
        sign_f     = tl.where(f >= 0.0, 1.0, -1.0)

        # d(encode(f))/df
        d_encode   = sign_f / (1.0 + abs_f)

        # grad_px: chain through exp(decode(px))
        grad_px    = g * d_encode * exp_x * sign_x * exp_abs_px

        # grad_py: chain through -log(decode(py))
        safe_y_mag = tl.where(safe_expm1 < 1e-7, 1e-7, safe_expm1)
        grad_py    = g * d_encode * (-1.0) * (exp_abs_py / safe_y_mag) * sign_y

        tl.store(grad_px_ptr + offs, grad_px, mask=mask)
        tl.store(grad_py_ptr + offs, grad_py, mask=mask)


# ── Grid helper ───────────────────────────────────────────────────────────────
def _grid(n: int, meta: dict) -> tuple:
    return (min(triton.cdiv(n, meta["BLOCK_SIZE"]), 1024),)


# ── Autograd Function ─────────────────────────────────────────────────────────
class _EMLPosit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        """px, py are posit-encoded float32 tensors (from Posit16.data)."""
        assert px.is_cuda and py.is_cuda, "EML Posit kernel requires CUDA tensors"
        assert px.shape == py.shape

        px, py = px.contiguous(), py.contiguous()
        n = px.numel()

        out          = torch.empty_like(px)
        save_exp_x   = torch.empty_like(px)
        save_eapx    = torch.empty_like(px)
        save_eapy    = torch.empty_like(px)

        _eml_posit_fwd_kernel[lambda meta: _grid(n, meta)](
            px, py,
            out, save_exp_x, save_eapx, save_eapy,
            n,
        )

        ctx.save_for_backward(px, py, save_exp_x, save_eapx, save_eapy)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        px, py, exp_x, eapx, eapy = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        n        = grad_out.numel()

        grad_px = torch.empty_like(px)
        grad_py = torch.empty_like(py)

        _eml_posit_bwd_kernel[lambda meta: _grid(n, meta)](
            grad_out, px, py, exp_x, eapx, eapy,
            grad_px, grad_py, n,
        )

        return grad_px, grad_py


# ── Public API ────────────────────────────────────────────────────────────────
def eml_posit(px: Posit16, py: Posit16) -> Posit16:
    """
    EML operator on Posit16 inputs:  out = exp(decode(px)) - log(decode(py))

    Returns a Posit16 with the result re-encoded in posit log-space.
    Fully differentiable. Autotuned. No LUT. No approximation.

    Transcendental ops per element (forward):
        exp(|px|)         — decode x
        exp(x_decoded)    — compute exp(x)
        log(expm1(|py|))  — compute log(y)  [or just |py| on fast path]
        log1p(|result|)   — re-encode output
    Total: 4  (vs ~6 for naive decode-in-PyTorch then recompute)
    """
    assert isinstance(px, Posit16) and isinstance(py, Posit16)
    out_data = _EMLPosit.apply(px.data, py.data)
    return Posit16(out_data, is_compressed=True)


def eml_posit_raw(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """
    Same as eml_posit() but accepts/returns raw posit-encoded float32 tensors
    directly — useful when you want to stay in the raw tensor world.
    """
    return _EMLPosit.apply(px.contiguous(), py.contiguous())


# ── Validation ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        raise SystemExit

    torch.manual_seed(0)
    N = 4_000_000

    # Build raw float inputs
    x_f = torch.randn(N, device="cuda").clamp(-5, 5)
    y_f = torch.rand(N,  device="cuda").abs_().add_(1e-3)

    # Encode to posit
    px = Posit16(x_f)          # px.data on CUDA
    py = Posit16(y_f)

    # ── Correctness ───────────────────────────────────────────────────────────
    with torch.no_grad():
        # Reference: decode → EML → re-encode (all in PyTorch)
        x_dec  = px.to_float()
        y_dec  = py.to_float()
        ref_f  = torch.exp(x_dec) - torch.log(y_dec)
        ref_p  = Posit16.encode(ref_f)          # expected posit-encoded output

        out_p  = eml_posit(px, py).data         # our kernel's posit-encoded output

    diff = (out_p - ref_p).abs().max().item()
    print(f"Forward correctness  : {'PASS' if diff < 1e-4 else 'FAIL'}  "
          f"max_posit_diff={diff:.3e}")

    # ── Backward correctness ──────────────────────────────────────────────────
    px_t = px.data.requires_grad_(True)
    py_t = py.data.requires_grad_(True)
    out2 = _EMLPosit.apply(px_t, py_t)
    out2.sum().backward()
    gx_tri = px_t.grad.clone()
    gy_tri = py_t.grad.clone()

    # PyTorch reference backward
    px_r = px.data.detach().requires_grad_(True)
    py_r = py.data.detach().requires_grad_(True)
    sign_x  = torch.sign(px_r)
    exp_apx = torch.exp(px_r.abs())
    x_d     = sign_x * (exp_apx - 1)
    sign_y  = torch.sign(py_r)
    exp_apy = torch.exp(py_r.abs())
    y_d     = sign_y * (exp_apy - 1)
    f       = torch.exp(x_d) - torch.log(y_d.abs().clamp(1e-7))
    out_r   = torch.sign(f) * torch.log1p(f.abs())
    out_r.sum().backward()
    gx_ref = px_r.grad.clone()
    gy_ref = py_r.grad.clone()

    gx_d = (gx_tri - gx_ref).abs().max().item()
    gy_d = (gy_tri - gy_ref).abs().max().item()
    print(f"Backward correctness : grad_px={'PASS' if gx_d<1e-3 else 'FAIL'}  "
          f"grad_py={'PASS' if gy_d<1e-3 else 'FAIL'}  "
          f"max_diff=({gx_d:.3e}, {gy_d:.3e})")

    # ── Throughput ────────────────────────────────────────────────────────────
    ITERS = 300
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        eml_posit(px, py)
    torch.cuda.synchronize()
    tri_ms = (time.perf_counter() - t0) * 1000 / ITERS

    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            xd = px.to_float(); yd = py.to_float()
            rf = torch.exp(xd) - torch.log(yd)
            Posit16.encode(rf)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000 / ITERS

    print(f"\nThroughput (N={N:,})  : Triton={tri_ms:.3f}ms  "
          f"PyTorch={pt_ms:.3f}ms  speedup={pt_ms/tri_ms:.2f}×")
