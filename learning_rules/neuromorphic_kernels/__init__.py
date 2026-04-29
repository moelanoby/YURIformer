import torch
import torch.nn as nn
import torch.nn.functional as F

from .OSTL import (
    compute_ostl_traces_triton, compute_ostl_traces_numba,
    OSTL_Function, manual_train_step_ostl,
)
from .OSTTP import (
    compute_osttp_traces_triton, compute_osttp_target_projection,
    compute_osttp_traces_numba, compute_osttp_target_projection_numba,
    OSTTP_Function, manual_train_step_osttp,
)

__all__ = [
    # Low-level trace kernels
    'compute_ostl_traces_triton',
    'compute_ostl_traces_numba',
    'compute_osttp_traces_triton',
    'compute_osttp_target_projection',
    'compute_osttp_traces_numba',
    'compute_osttp_target_projection_numba',
    # High-level autograd functions (online local learning)
    'OSTL_Function',
    'OSTTP_Function',
    # Modular manual training steps
    'manual_train_step_ostl',
    'manual_train_step_osttp',
]
