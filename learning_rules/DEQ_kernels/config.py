"""
Configuration & Factory for DEQ Solvers.

Provides dataclass-based configs for each solver so users can
declaratively specify solver settings and instantiate them via
``SolverFactory.create(config, f)``.

Example
-------
>>> from DEQ_kernels import HybridConfig, SolverFactory
>>>
>>> config = HybridConfig(
...     max_iter=30,
...     tol=1e-4,
...     pjwr_iters=5,
...     anderson_m=8,
...     broyden_memory=10,
... )
>>> solver = SolverFactory.create(config, my_layer)
>>> z_star = solver.solve(x)
"""

from dataclasses import dataclass, field
from typing import Optional

from .pjwr import ParallelJacobiWaveformSolver
from .anderson import AndersonSolver
from .broyden import BroydenSolver
from .hybrid import HybridAndersonBroydenSolver


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PJWRConfig:
    """Configuration for ParallelJacobiWaveformSolver."""
    max_iter: int = 40
    tol: float = 1e-3
    use_shanks: bool = True
    numeric_mode: str = 'float'   # 'float' or 'posit16'


@dataclass
class AndersonConfig:
    """Configuration for AndersonSolver (Sketched Anderson Acceleration)."""
    max_iter: int = 50
    tol: float = 1e-5
    m: int = 5                    # mixing window
    beta: float = 1.0             # damping
    sketch_size: Optional[int] = 256  # None = no sketching


@dataclass
class BroydenConfig:
    """Configuration for BroydenSolver (Limited-Memory Broyden)."""
    max_iter: int = 50
    tol: float = 1e-5
    memory: int = 20              # rank-1 update buffer size


@dataclass
class HybridConfig:
    """
    Configuration for HybridAndersonBroydenSolver.
    
    Combines PJWR warm-up → Anderson → Broyden in a single iteration budget.
    """
    max_iter: int = 50
    tol: float = 1e-5
    # PJWR phase
    pjwr_iters: int = 8
    use_shanks: bool = True
    # Anderson phase
    anderson_m: int = 5
    anderson_beta: float = 1.0
    n_blocks: int = 4
    sketch_size: Optional[int] = 256
    # Broyden phase
    broyden_memory: int = 15
    switch_tol: float = 1e-2      # residual to trigger Broyden switch


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class SolverFactory:
    """
    Build a solver instance from a config object.

    Usage
    -----
    >>> solver = SolverFactory.create(HybridConfig(max_iter=30), my_layer)
    """

    _registry = {
        PJWRConfig: ParallelJacobiWaveformSolver,
        AndersonConfig: AndersonSolver,
        BroydenConfig: BroydenSolver,
        HybridConfig: HybridAndersonBroydenSolver,
    }

    @classmethod
    def create(cls, config, f):
        """
        Instantiate a solver from its config.

        Parameters
        ----------
        config : PJWRConfig | AndersonConfig | BroydenConfig | HybridConfig
            Solver configuration dataclass.
        f : callable
            The equilibrium map ``f(z, x)`` (or an ``nn.Module``).

        Returns
        -------
        Solver instance with a ``.solve(x, z_init)`` method.

        Raises
        ------
        ValueError
            If ``config`` type is not recognized.
        """
        solver_cls = cls._registry.get(type(config))
        if solver_cls is None:
            supported = ', '.join(c.__name__ for c in cls._registry)
            raise ValueError(
                f"Unknown config type {type(config).__name__}. "
                f"Supported: {supported}"
            )

        # Convert dataclass fields to kwargs
        from dataclasses import asdict
        kwargs = asdict(config)
        return solver_cls(f=f, **kwargs)

    @classmethod
    def register(cls, config_cls, solver_cls):
        """
        Register a custom (config, solver) pair so SolverFactory can
        build your solver too.

        Parameters
        ----------
        config_cls : type
            A dataclass type.
        solver_cls : type
            A solver class whose ``__init__`` accepts ``f`` plus the
            config's fields as keyword arguments.
        """
        cls._registry[config_cls] = solver_cls
