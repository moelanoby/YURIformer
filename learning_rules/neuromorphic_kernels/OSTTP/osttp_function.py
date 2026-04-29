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
