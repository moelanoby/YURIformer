"""
Implicit Differentiation for DEQ: DEQFunction + DEQModule.

Provides the backward pass machinery that makes DEQ solvers a
drop-in replacement for BPTT. Supports three backward modes:

- ``'phantom'``   : 0-step Neumann — instant O(1) backward (default)
- ``'neumann-1'`` : 1-step Neumann — one VJP, still very fast
- ``'ift'``       : Full IFT via adjoint Anderson solve (legacy, slow)
"""

import torch
import torch.nn as nn
from .anderson import AndersonSolver
from .hybrid import HybridAndersonBroydenSolver


class DEQFunction(torch.autograd.Function):
    """
    Custom autograd Function implementing implicit differentiation.

    Forward: runs the solver to find z* = f(z*, x).
    Backward: computes gradients via Phantom / Neumann / IFT.
    """

    @staticmethod
    def forward(ctx, f, x, solver, backward_mode, *params):
        with torch.no_grad():
            z_star = solver.solve(x)

        ctx.f = f
        ctx.x = x
        ctx.solver = solver
        ctx.backward_mode = backward_mode
        ctx.save_for_backward(z_star, *params)
        return z_star

    @staticmethod
    def backward(ctx, grad_z):
        z_star, *params = ctx.saved_tensors
        f = ctx.f
        x = ctx.x
        solver = ctx.solver
        mode = getattr(ctx, 'backward_mode', 'phantom')

        z_star = z_star.detach().requires_grad_(True)
        x_needs_grad = x.requires_grad
        x = x.detach().requires_grad_(True)

        if mode == 'phantom':
            # 0-step Neumann (Phantom Gradients): Instant backward pass (0 VJPs)
            g_star = grad_z

        elif mode == 'neumann-1':
            # 1-step Neumann: Ultra-fast backward pass (1 VJP)
            with torch.enable_grad():
                y = f(z_star, x)
                vjp = torch.autograd.grad(y, z_star, grad_z, retain_graph=True)[0]
            g_star = grad_z + vjp

        else:
            # Full IFT Solver (Legacy Anderson, Slow)
            def adjoint_f(g, grad_z_val):
                with torch.enable_grad():
                    y = f(z_star, grad_z_val)
                    vjp = torch.autograd.grad(y, z_star, g, retain_graph=True)[0]
                return vjp + grad_z

            with torch.no_grad():
                adjoint_solver = AndersonSolver(
                    adjoint_f, max_iter=10, tol=solver.tol, m=5, beta=1.0
                )
                g_star = adjoint_solver.solve(x, z_init=grad_z)

        # Compute parameter and input gradients
        with torch.enable_grad():
            y = f(z_star, x)
            grads = torch.autograd.grad(y, params, g_star,
                                        retain_graph=True, allow_unused=True)
            grad_x = torch.autograd.grad(y, x, g_star,
                                         retain_graph=True, allow_unused=True)[0]

        # Only propagate input gradient if the original input required it
        if not x_needs_grad:
            grad_x = None

        return (None, grad_x, None, None) + grads


class DEQModule(nn.Module):
    """
    General-purpose DEQ wrapper — drop-in replacement for BPTT.

    Wraps any equilibrium layer and runs it to convergence using the
    configured solver, with implicit differentiation on the backward pass.

    Parameters
    ----------
    layer : nn.Module
        The equilibrium layer ``f(z, x)`` to solve.
    solver : solver instance, optional
        Any solver with a ``.solve(x, z_init)`` method. Defaults to
        ``HybridAndersonBroydenSolver``.
    backward_mode : str
        ``'phantom'`` | ``'neumann-1'`` | ``'ift'``
    **solver_kwargs
        Forwarded to the default Hybrid solver if ``solver`` is None.

    Example
    -------
    >>> deq = DEQModule(my_layer, backward_mode='phantom')
    >>> z_star = deq(x)
    """

    def __init__(self, layer, solver=None, backward_mode='phantom', **solver_kwargs):
        super().__init__()
        self.layer = layer
        self.backward_mode = backward_mode
        if solver is None:
            self.solver = HybridAndersonBroydenSolver(self.layer, **solver_kwargs)
        else:
            self.solver = solver

    def forward(self, x, z_init=None):
        params = tuple(self.layer.parameters())
        return DEQFunction.apply(
            self.layer, x, self.solver, self.backward_mode, *params
        )
