import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_osttp_traces_numba(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Computes eligibility traces for OSTTP (Online Spatio-Temporal Target Projection).
    Optimized with Numba for fast execution on CPU.
    
    Args:
        x: Input array of shape (seq_len, batch_size, hidden_size)
        decay: Temporal decay factor (0 < decay < 1)
    Returns:
        e: Eligibility traces of the same shape as x
    """
    seq_len, batch_size, hidden_size = x.shape
    e = np.empty_like(x)
    
    for b in prange(batch_size):
        for h in range(hidden_size):
            trace = 0.0
            for t in range(seq_len):
                trace = trace * decay + x[t, b, h]
                e[t, b, h] = trace
                
    return e

@njit(parallel=True, fastmath=True)
def compute_osttp_target_projection_numba(error_signal: np.ndarray, random_projection_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the target projection for the learning signal in OSTTP.
    
    Args:
        error_signal: shape (batch_size, out_features)
        random_projection_matrix: shape (out_features, hidden_size)
    Returns:
        Projected error signal: shape (batch_size, hidden_size)
    """
    # Simple parallel dot product implementation
    batch_size, out_features = error_signal.shape
    _, hidden_size = random_projection_matrix.shape
    
    projected = np.zeros((batch_size, hidden_size))
    
    for b in prange(batch_size):
        for h in range(hidden_size):
            val = 0.0
            for o in range(out_features):
                val += error_signal[b, o] * random_projection_matrix[o, h]
            projected[b, h] = val
            
    return projected
