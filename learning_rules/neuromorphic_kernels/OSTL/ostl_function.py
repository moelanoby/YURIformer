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


# =============================================================================
# Shared helpers
# =============================================================================

def _softmax_grad(logits, targets):
    """Cross-entropy + softmax gradient w.r.t. logits, shape (B, C)."""
    g = torch.softmax(logits, dim=1)
    g[torch.arange(targets.size(0)), targets] -= 1.0
    g /= targets.size(0)
    return g


def _clip_grads(params, max_norm=1.0):
    """In-place gradient clipping over an iterable of parameters."""
    total_sq = sum(
        p.grad.data.norm(2).item() ** 2
        for p in params if p.grad is not None
    )
    norm = total_sq ** 0.5
    coef = max_norm / (norm + 1e-6)
    if coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(coef)


def _update_cell_grads(cell, error, z_trace, h_trace):
    """
    Write OSTL weight gradients for one RecurrentCell.

        ΔWx  = error.T @ z_trace
        ΔWz  = error.T @ h_trace
        Δb   = error.sum(0)
    """
    cell.Wx.weight.grad = error.t() @ z_trace
    cell.Wz.weight.grad = error.t() @ h_trace
    if cell.Wz.bias is not None:
        cell.Wz.bias.grad = error.sum(0)


# =============================================================================
# Manual training step
# =============================================================================

def manual_train_step_ostl(model, x, y, optimizer, max_norm=1.0):
    """
    Full online training step for a Deep_OSTL_Model — no BPTT, no autograd
    on the recurrent cells.

    Flow:
        1. local_forward → logits, per-layer traces
        2. Softmax gradient → error signal
        3. Head & proj_in updated via standard outer-product rule
        4. Each cell updated via OSTL three-factor rule (error × trace)
        5. Gradient clip → optimizer.step()

    Args:
        model     – Deep_OSTL_Model with a .local_forward() method
        x, y      – batch inputs / labels (already on device)
        optimizer – any torch.optim optimiser
        max_norm  – gradient clip threshold (default 1.0)

    Returns:
        loss value (float)
    """
    logits, traces, x_input, z_final = model.local_forward(x)
    loss = torch.nn.functional.cross_entropy(logits, y)

    optimizer.zero_grad()

    with torch.no_grad():
        # 1. Gradient of cross-entropy loss w.r.t. logits
        error = _softmax_grad(logits, y)

        # 2. Head layer
        model.head.weight.grad = error.t() @ z_final
        if model.head.bias is not None:
            model.head.bias.grad = error.sum(0)

        # 3. Propagate error back to hidden dim (head weight transpose)
        error = error @ model.head.weight  # (B, D)

        # 4. Per-layer OSTL updates; error broadcast (unchanged) to each layer
        for z_trace, h_trace, cell in reversed(traces):
            _update_cell_grads(cell, error, z_trace, h_trace)
            # error stays unchanged → same signal for all layers (Bypass)

        # 5. Input projection
        model.proj_in.weight.grad = error.t() @ x_input
        if model.proj_in.bias is not None:
            model.proj_in.bias.grad = error.sum(0)

    _clip_grads(model.parameters(), max_norm)
    optimizer.step()
    return loss.item()
