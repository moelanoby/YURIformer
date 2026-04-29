import torch
import torch.nn as nn
import torch.nn.functional as F

from .OSTL import compute_ostl_traces_triton, compute_ostl_traces_numba
from .OSTTP import compute_osttp_traces_triton, compute_osttp_target_projection, compute_osttp_traces_numba, compute_osttp_target_projection_numba
from .ostl_function import OSTL_Function
from .osttp_function import OSTTP_Function

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
]
