"""
OSTL_Function — Online Spatio-Temporal Learning
================================================
Implements the three-factor local learning rule for ANNs:

    e_h[t] = decay * e_h[t-1] + h[t]      (hidden eligibility trace)
    e_z     = z * sum_decay                 (input eligibility trace, constant input)
    ΔWx    += error  @ e_z.T
    ΔWz    += error  @ e_h.T

Crucially:
  - The entire recurrence runs under torch.no_grad() → no BPTT graph built.
  - Weight gradients are written manually; no autograd is used for the cell.
  - The incoming error signal is passed through unchanged (Error Broadcast /
    Bypass) so that outer layers (proj_in, head) can still be updated via
    standard gradient descent.
"""

import torch


class OSTL_Function(torch.autograd.Function):
    """
    Args (forward):
        z          – input from previous layer, shape (B, D)
        cell       – nn.Module implementing cell(h, z) → h_next
        n_steps    – number of recurrent steps
        decay      – eligibility trace decay (0 < decay < 1)

    Returns:
        h          – final hidden state, shape (B, D)
    """

    @staticmethod
    def forward(ctx, z, cell, n_steps, decay):
        ctx.cell    = cell
        ctx.n_steps = n_steps
        ctx.decay   = decay

        with torch.no_grad():
            h       = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)

            for _ in range(n_steps):
                h       = cell(h, z)
                h_trace = decay * h_trace + h   # e_h[t] = λ·e_h[t-1] + h[t]

            # Input is constant over the sequence; closed-form geometric sum
            sum_decay = (1.0 - decay ** n_steps) / (1.0 - decay)
            z_trace   = z * sum_decay

        ctx.save_for_backward(z, h_trace, z_trace)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output – incoming error signal, shape (B, D)

        Manual weight update (no autograd, no backprop through cell):
            ΔWx += error.T @ z_trace
            ΔWz += error.T @ h_trace
            Δb  += error.sum(0)

        Returns grad_output unchanged to upstream layers (Error Broadcast).
        """
        z, h_trace, z_trace = ctx.saved_tensors
        cell = ctx.cell

        with torch.no_grad():
            # --- Wx ---
            if cell.Wx.weight.grad is None:
                cell.Wx.weight.grad = torch.zeros_like(cell.Wx.weight)
            cell.Wx.weight.grad.add_(grad_output.t() @ z_trace)

            # --- Wz ---
            if cell.Wz.weight.grad is None:
                cell.Wz.weight.grad = torch.zeros_like(cell.Wz.weight)
            cell.Wz.weight.grad.add_(grad_output.t() @ h_trace)

            # --- bias ---
            if cell.Wz.bias is not None:
                if cell.Wz.bias.grad is None:
                    cell.Wz.bias.grad = torch.zeros_like(cell.Wz.bias)
                cell.Wz.bias.grad.add_(grad_output.sum(0))

        # Pass error signal through without modification (Bypass / Error Broadcast)
        return grad_output, None, None, None
