import torch
import triton
import triton.language as tl

@triton.jit
def _ostl_trace_kernel(
    x_ptr, e_ptr,
    seq_len, batch_size, hidden_size,
    decay,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr
):
    """
    Computes the eligibility trace for OSTL.
    e[t] = decay * e[t-1] + x[t]
    x: (seq_len, batch_size, hidden_size)
    e: (seq_len, batch_size, hidden_size)
    """
    # Grid is over batch and hidden dimensions
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    b_offs = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    h_offs = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    
    b_mask = b_offs < batch_size
    h_mask = h_offs < hidden_size
    
    # Offsets for the first timestep
    # x shape: (seq_len, batch_size, hidden_size)
    # strides: (batch_size * hidden_size, hidden_size, 1)
    stride_seq = batch_size * hidden_size
    stride_b = hidden_size
    stride_h = 1
    
    base_offs = b_offs[:, None] * stride_b + h_offs[None, :] * stride_h
    
    # Initialize trace accumulator
    trace = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)
    
    mask = b_mask[:, None] & h_mask[None, :]
    
    for t in range(seq_len):
        x_ptrs = x_ptr + t * stride_seq + base_offs
        e_ptrs = e_ptr + t * stride_seq + base_offs
        
        x_val = tl.load(x_ptrs, mask=mask, other=0.0)
        trace = trace * decay + x_val
        
        tl.store(e_ptrs, trace, mask=mask)

def compute_ostl_traces_triton(x: torch.Tensor, decay: float):
    """
    Computes eligibility traces for continuous-valued ANNs using OSTL.
    Args:
        x: Input tensor of shape (seq_len, batch_size, hidden_size)
        decay: Temporal decay factor (0 < decay < 1)
    Returns:
        e: Eligibility traces of the same shape as x
    """
    seq_len, batch_size, hidden_size = x.shape
    e = torch.empty_like(x)
    
    BLOCK_SIZE_B = 16
    BLOCK_SIZE_H = 64
    
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(hidden_size, BLOCK_SIZE_H))
    
    _ostl_trace_kernel[grid](
        x, e,
        seq_len, batch_size, hidden_size,
        decay,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H
    )
    
    return e
