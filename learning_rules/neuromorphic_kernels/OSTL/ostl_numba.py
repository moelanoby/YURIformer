import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_ostl_traces_numba(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Computes eligibility traces for continuous-valued ANNs using OSTL.
    Optimized with Numba for fast execution on CPU.
    
    Args:
        x: Input array of shape (seq_len, batch_size, hidden_size)
        decay: Temporal decay factor (0 < decay < 1)
    Returns:
        e: Eligibility traces of the same shape as x
    """
    seq_len, batch_size, hidden_size = x.shape
    e = np.empty_like(x)
    
    # Parallelize over batch and hidden dimensions
    for b in prange(batch_size):
        for h in range(hidden_size):
            trace = 0.0
            for t in range(seq_len):
                trace = trace * decay + x[t, b, h]
                e[t, b, h] = trace
                
    return e
