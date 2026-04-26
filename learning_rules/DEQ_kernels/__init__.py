"""
DEQ Solvers Library
═══════════════════

A modular, configurable library of Deep Equilibrium (DEQ) solvers
designed to replace BPTT with implicit differentiation.

Solvers
-------
- ParallelJacobiWaveformSolver : GPU-parallel fixed-point iteration + Shanks acceleration
- AndersonSolver               : Sketched Anderson Acceleration with ring buffers
- BroydenSolver                : Limited-memory Broyden's method with batched updates
- HybridAndersonBroydenSolver  : 3-phase PJWR → Anderson → Broyden pipeline

Implicit Differentiation
------------------------
- DEQFunction : torch.autograd.Function with Phantom / Neumann / IFT backward
- DEQModule   : nn.Module wrapper for any equilibrium layer

Configuration
-------------
- PJWRConfig, AndersonConfig, BroydenConfig, HybridConfig
- SolverFactory.create(config, f) — build a solver from a config object

Utilities
---------
- ShanksAccelerator       : Aitken delta-squared sequence acceleration
- jacobian_spectral_norm  : Power-iteration spectral regularization

Quick Start
-----------
>>> from DEQ_kernels import DEQModule, HybridConfig, SolverFactory
>>> config = HybridConfig(max_iter=30, tol=1e-4)
>>> solver = SolverFactory.create(config, my_layer)
>>> deq = DEQModule(my_layer, solver=solver, backward_mode='phantom')
>>> z_star = deq(x)
"""

__version__ = "3.0.0"

# ── Solvers ──────────────────────────────────────────────────────────────────
from .pjwr import ParallelJacobiWaveformSolver
from .anderson import AndersonSolver
from .broyden import BroydenSolver
from .hybrid import HybridAndersonBroydenSolver

# ── Implicit Differentiation ─────────────────────────────────────────────────
from .implicit_diff import DEQFunction, DEQModule

# ── Utilities ────────────────────────────────────────────────────────────────
from .accelerators import ShanksAccelerator
from .regularization import jacobian_spectral_norm

# ── Configuration & Factory ──────────────────────────────────────────────────
from .config import (
    PJWRConfig,
    AndersonConfig,
    BroydenConfig,
    HybridConfig,
    SolverFactory,
)

# ── Triton Kernels ───────────────────────────────────────────────────────────
from .triton_kernels import deq_fixed_point_kernel

# ── Posit Numerics ───────────────────────────────────────────────────────────
import sys, os
try:
    from numeric_kernels.posit import Posit16
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from numeric_kernels.posit import Posit16

__all__ = [
    # Solvers
    "ParallelJacobiWaveformSolver",
    "AndersonSolver",
    "BroydenSolver",
    "HybridAndersonBroydenSolver",
    # Implicit Diff
    "DEQFunction",
    "DEQModule",
    # Utilities
    "ShanksAccelerator",
    "jacobian_spectral_norm",
    # Config
    "PJWRConfig",
    "AndersonConfig",
    "BroydenConfig",
    "HybridConfig",
    "SolverFactory",
    # Kernels
    "deq_fixed_point_kernel",
    # Numerics
    "Posit16",
]
