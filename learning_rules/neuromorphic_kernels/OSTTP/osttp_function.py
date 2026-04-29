"""
OSTTP_Function — Online Spatio-Temporal Learning with Target Projection
=======================================================================
Extends OSTL by replacing spatial backpropagation (weight.T @ error) with
Direct Random Target Projection (DRTP):

    projected_error = error @ B          (B is a fixed random matrix)
    ΔWx += projected_error.T @ e_z
    ΔWz += projected_error.T @ e_h

This removes the weight-transport / feedback alignment problem and makes
spatial credit assignment fully local.  No backpropagation through layer
weights ever occurs.

The incoming error is still forwarded unchanged (Bypass) so that the
classification head and input projection can be trained normally.
"""

import torch


class OSTTP_Function(torch.autograd.Function):
    """
    Args (forward):
        z           – input from previous layer, shape (B, D)
        cell        – nn.Module implementing cell(h, z) → h_next
        random_proj – fixed random matrix, shape (D, D)
        n_steps     – number of recurrent steps
        decay       – eligibility trace decay (0 < decay < 1)

    Returns:
        h           – final hidden state, shape (B, D)
    """

    @staticmethod
    def forward(ctx, z, cell, random_proj, n_steps, decay):
        ctx.cell        = cell
        ctx.random_proj = random_proj
        ctx.n_steps     = n_steps
        ctx.decay       = decay

        with torch.no_grad():
            h       = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)

            for _ in range(n_steps):
                h       = cell(h, z)
                h_trace = decay * h_trace + h   # e_h[t] = λ·e_h[t-1] + h[t]

            sum_decay = (1.0 - decay ** n_steps) / (1.0 - decay)
            z_trace   = z * sum_decay

        ctx.save_for_backward(z, h_trace, z_trace)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output – incoming error signal, shape (B, D)

        Target Projection:
            projected_error = error @ B          (no weight transpose!)
            ΔWx += projected_error.T @ z_trace
            ΔWz += projected_error.T @ h_trace
            Δb  += projected_error.sum(0)

        Returns grad_output unchanged (Bypass).
        """
        z, h_trace, z_trace = ctx.saved_tensors
        cell        = ctx.cell
        random_proj = ctx.random_proj

        # Project global error locally — no weight-transport needed
        projected_error = grad_output @ random_proj   # (B, D)

        with torch.no_grad():
            # --- Wx ---
            if cell.Wx.weight.grad is None:
                cell.Wx.weight.grad = torch.zeros_like(cell.Wx.weight)
            cell.Wx.weight.grad.add_(projected_error.t() @ z_trace)

            # --- Wz ---
            if cell.Wz.weight.grad is None:
                cell.Wz.weight.grad = torch.zeros_like(cell.Wz.weight)
            cell.Wz.weight.grad.add_(projected_error.t() @ h_trace)

            # --- bias ---
            if cell.Wz.bias is not None:
                if cell.Wz.bias.grad is None:
                    cell.Wz.bias.grad = torch.zeros_like(cell.Wz.bias)
                cell.Wz.bias.grad.add_(projected_error.sum(0))

        # Pass error signal through without modification (Bypass / Error Broadcast)
        return grad_output, None, None, None, None


# =============================================================================
# Shared helpers  (mirrored from ostl_function for self-containment)
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


def _update_cell_grads_projected(cell, projected_error, z_trace, h_trace):
    """
    Write OSTTP weight gradients for one RecurrentCell.

        ΔWx  = projected_error.T @ z_trace
        ΔWz  = projected_error.T @ h_trace
        Δb   = projected_error.sum(0)
    """
    cell.Wx.weight.grad = projected_error.t() @ z_trace
    cell.Wz.weight.grad = projected_error.t() @ h_trace
    if cell.Wz.bias is not None:
        cell.Wz.bias.grad = projected_error.sum(0)


# =============================================================================
# Manual training step
# =============================================================================

def manual_train_step_osttp(model, x, y, optimizer, max_norm=1.0):
    """
    Full online training step for a Deep_OSTTP_Model — no BPTT, no weight
    transport.  Each cell receives a *locally projected* error signal via a
    fixed random matrix (Direct Random Target Projection).

    Flow:
        1. local_forward → logits, per-layer traces + random projections
        2. Softmax gradient → global error
        3. Head & proj_in updated via standard outer-product rule
        4. Each cell updated with  projected_error = error @ B  (DRTP)
        5. Gradient clip → optimizer.step()

    Args:
        model     – Deep_OSTTP_Model with a .local_forward() method
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

        # 3. Propagate error back to hidden dim
        error = error @ model.head.weight  # (B, D)

        # 4. Per-layer OSTTP updates with target projection
        for z_trace, h_trace, cell, rand_proj in reversed(traces):
            projected_error = error @ rand_proj  # local signal — no weight transport
            _update_cell_grads_projected(cell, projected_error, z_trace, h_trace)
            # error stays unchanged → same global signal bypassed to previous layer

        # 5. Input projection
        model.proj_in.weight.grad = error.t() @ x_input
        if model.proj_in.bias is not None:
            model.proj_in.bias.grad = error.sum(0)

    _clip_grads(model.parameters(), max_norm)
    optimizer.step()
    return loss.item()
