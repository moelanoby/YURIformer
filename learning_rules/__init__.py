"""
learning_rules
==============

Top-level package for YURIformer's learning rule library.
Exposes two sub-libraries as clean, importable namespaces:

Sub-packages
------------
- DEQ_kernels       : Deep Equilibrium solvers (Hybrid, Anderson, Broyden, PJWR)
- neuromorphic_kernels : Online local learning rules (OSTL, OSTTP) + Triton/Numba kernels

Quick Start
-----------
>>> # DEQ (implicit infinite-depth, O(1) memory)
>>> from learning_rules.DEQ_kernels import DEQModule, HybridConfig, SolverFactory

>>> # OSTL / OSTTP (online local learning, no BPTT)
>>> from learning_rules.neuromorphic_kernels import (
...     OSTL_Function, manual_train_step_ostl,
...     OSTTP_Function, manual_train_step_osttp,
... )
"""

# ── DEQ solvers ───────────────────────────────────────────────────────────────
from .DEQ_kernels import (
    # Solvers
    ParallelJacobiWaveformSolver,
    AndersonSolver,
    BroydenSolver,
    HybridAndersonBroydenSolver,
    # Implicit diff module
    DEQFunction,
    DEQModule,
    # Config / Factory
    PJWRConfig,
    AndersonConfig,
    BroydenConfig,
    HybridConfig,
    SolverFactory,
    # Utilities
    ShanksAccelerator,
    jacobian_spectral_norm,
)

# ── Neuromorphic / Online learning rules ─────────────────────────────────────
from .neuromorphic_kernels import (
    # Low-level trace kernels (Triton + Numba)
    compute_ostl_traces_triton,
    compute_ostl_traces_numba,
    compute_osttp_traces_triton,
    compute_osttp_target_projection,
    compute_osttp_traces_numba,
    compute_osttp_target_projection_numba,
    # High-level autograd functions
    OSTL_Function,
    OSTTP_Function,
    # Self-contained modular training steps
    manual_train_step_ostl,
    manual_train_step_osttp,
)

__all__ = [
    # ── DEQ ──────────────────────────────────────────────────────────────────
    "ParallelJacobiWaveformSolver",
    "AndersonSolver",
    "BroydenSolver",
    "HybridAndersonBroydenSolver",
    "DEQFunction",
    "DEQModule",
    "PJWRConfig",
    "AndersonConfig",
    "BroydenConfig",
    "HybridConfig",
    "SolverFactory",
    "ShanksAccelerator",
    "jacobian_spectral_norm",
    # ── Neuromorphic ─────────────────────────────────────────────────────────
    "compute_ostl_traces_triton",
    "compute_ostl_traces_numba",
    "compute_osttp_traces_triton",
    "compute_osttp_target_projection",
    "compute_osttp_traces_numba",
    "compute_osttp_target_projection_numba",
    "OSTL_Function",
    "OSTTP_Function",
    "manual_train_step_ostl",
    "manual_train_step_osttp",
]
