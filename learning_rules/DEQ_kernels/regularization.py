"""
Regularization utilities for DEQ training stability.
"""

import torch
import torch.nn.functional as F


def jacobian_spectral_norm(f, z_star, x, n_power_iters=5, target=0.9):
    """
    Estimate and penalize the spectral norm of the Jacobian ∂f/∂z at the
    equilibrium point z*.

    Uses power iteration on J^T J to approximate the largest singular value
    σ_max, then returns a penalty ReLU(σ_max - target)² that encourages the
    fixed-point map to be a contraction.

    Parameters
    ----------
    f : callable
        The equilibrium layer  f(z, x).
    z_star : Tensor
        The equilibrium solution (detached; gradients will be re-enabled).
    x : Tensor
        The input injection.
    n_power_iters : int
        Number of power-iteration steps (more → tighter estimate).
    target : float
        Desired upper bound on σ_max. Values < 1.0 enforce contraction.

    Returns
    -------
    Tensor (scalar)
        The penalty term to add to the training loss.

    Example
    -------
    >>> loss = task_loss + 0.1 * jacobian_spectral_norm(layer, z_star, x)
    """
    z = z_star.detach().requires_grad_(True)
    fz = f(z, x)

    # Power iteration to estimate largest singular value of J = ∂f/∂z
    v = torch.randn_like(z)
    v = v / (v.norm() + 1e-9)

    for _ in range(n_power_iters):
        # Jv via forward-mode AD (JVP)
        Jv = torch.autograd.grad(fz, z, v, create_graph=True, retain_graph=True)[0]
        # J^T(Jv) for power iteration on J^TJ
        JtJv = torch.autograd.grad(fz, z, Jv.detach(), retain_graph=True)[0]
        v = JtJv / (JtJv.norm() + 1e-9)

    # Final spectral norm estimate
    Jv = torch.autograd.grad(fz, z, v, create_graph=True, retain_graph=True)[0]
    sigma = Jv.norm()

    return F.relu(sigma - target) ** 2
