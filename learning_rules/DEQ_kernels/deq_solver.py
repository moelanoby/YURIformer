"""
Backward-compatibility shim.

All solver code has been split into individual modules under the
``DEQ_kernels`` package. This file re-exports every public name so that
existing ``from deq_solver import ...`` statements continue to work.

For new code, prefer importing from the package directly::

    from DEQ_kernels import DEQModule, AndersonSolver, HybridConfig, SolverFactory
"""

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
from .config import PJWRConfig, AndersonConfig, BroydenConfig, HybridConfig, SolverFactory

# ── Triton Kernels ───────────────────────────────────────────────────────────
from .triton_kernels import deq_fixed_point_kernel

# ── Posit Numerics ───────────────────────────────────────────────────────────
import sys, os
try:
    from numeric_kernels.posit import Posit16
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from numeric_kernels.posit import Posit16


if __name__ == "__main__":
    print("🚀 DEQ Solvers Library v3.0 — Modular Architecture")
    print("=" * 55)
    print()
    print("  Individual Solvers:")
    print("    • ParallelJacobiWaveformSolver  (pjwr.py)")
    print("    • AndersonSolver                (anderson.py)")
    print("    • BroydenSolver                 (broyden.py)")
    print("    • HybridAndersonBroydenSolver   (hybrid.py)")
    print()
    print("  Configuration:")
    print("    • PJWRConfig / AndersonConfig / BroydenConfig / HybridConfig")
    print("    • SolverFactory.create(config, f)")
    print()
    print("  Implicit Differentiation:")
    print("    • DEQFunction  (autograd.Function)")
    print("    • DEQModule    (nn.Module wrapper)")
    print()
    print("  Utilities:")
    print("    • ShanksAccelerator")
    print("    • jacobian_spectral_norm()")
    print()
    print("Ready to replace BPTT. ⚡")
