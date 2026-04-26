"""
Parallel Jacobi Waveform Relaxation (PJWR) Solver.

Optimized for solving equilibrium points across temporal dimensions or
decoupled subsystems in parallel. Unlike Anderson/Broyden which require
sequential history-dependent updates, PJWR updates all components
simultaneously — making it ideal for GPU-parallel workloads.

Trade-off: Linear convergence rate, but massively parallel.
Use when parallelism matters more than per-iteration convergence speed.
"""

import torch
from .accelerators import ShanksAccelerator
import sys, os
try:
    from numeric_kernels.posit import Posit16
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from numeric_kernels.posit import Posit16


class ParallelJacobiWaveformSolver:
    """
    GPU-parallel fixed-point iteration with optional Shanks acceleration
    and Posit16 numerics.

    Parameters
    ----------
    f : callable
        The equilibrium map ``f(z, x)`` whose fixed point ``z* = f(z*, x)``
        is sought.
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on the relative residual norm.
    use_shanks : bool
        Whether to apply Aitken delta-squared acceleration every iteration.
    numeric_mode : str
        ``'float'`` (default) or ``'posit16'`` for log-space emulated numerics.

    Example
    -------
    >>> solver = ParallelJacobiWaveformSolver(my_layer, max_iter=40, tol=1e-3)
    >>> z_star = solver.solve(x)
    """

    def __init__(self, f, max_iter=40, tol=1e-3, use_shanks=True, numeric_mode='float'):
        self.f = f
        self.max_iter = max_iter
        self.tol = tol
        self.use_shanks = use_shanks
        self.shanks = ShanksAccelerator()
        self.numeric_mode = numeric_mode

    def solve(self, x, z_init=None):
        """
        Run PJWR iteration to find the fixed point of ``f``.

        Parameters
        ----------
        x : Tensor
            Input injection.
        z_init : Tensor, optional
            Initial guess for the fixed point. Defaults to zeros.

        Returns
        -------
        Tensor
            Approximate fixed point z*.
        """
        z = z_init if z_init is not None else torch.zeros_like(x)

        if self.numeric_mode == 'posit16':
            z = Posit16(z)

        z_history = [z]

        for i in range(self.max_iter):
            pump = min(1.0, i / max(1, self.max_iter // 2))
            z_f = z.to_float() if isinstance(z, Posit16) else z
            z_next_f = self.f(z_f, x) * pump
            z_next = Posit16(z_next_f) if self.numeric_mode == 'posit16' else z_next_f

            diff_vec = z_next - z
            diff = diff_vec.norm() / (z_next.norm() + 1e-9)

            z = z_next
            z_history.append(z)

            if self.use_shanks and len(z_history) >= 3:
                z = self.shanks.accelerate(z_history)
                z_history[-1] = z

            if diff < self.tol:
                break

        if i == self.max_iter - 1:
            print(f"  [PJWR] Warning: Max iterations reached (diff={diff:.4e}).")

        return z.to_float() if isinstance(z, Posit16) else z
