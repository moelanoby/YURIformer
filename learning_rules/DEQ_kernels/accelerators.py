"""
Sequence Accelerators for fixed-point iteration.
"""

import torch


class ShanksAccelerator:
    """
    Shanks Transformation (Aitken's Delta-Squared process).
    Accelerates the convergence of a sequence of hidden states.
    Effective for geometric or near-geometric convergence patterns.

    Key advantage: Fully element-wise — naturally parallelizable across
    all dimensions with zero cross-dimensional dependencies.

    Usage
    -----
    >>> acc = ShanksAccelerator()
    >>> z_history = [z0, z1, z2]
    >>> z_acc = acc.accelerate(z_history)
    """

    @staticmethod
    def accelerate(z_history):
        """
        Apply Aitken delta-squared acceleration to the last 3 iterates.

        Parameters
        ----------
        z_history : list[Tensor]
            At least 3 consecutive iterates of the fixed-point map.

        Returns
        -------
        Tensor
            Accelerated estimate of the fixed point.
        """
        if len(z_history) < 3:
            return z_history[-1]

        z_k2, z_k1, z_k = z_history[-3:]

        num = (z_k - z_k1) ** 2
        den = z_k - 2 * z_k1 + z_k2

        eps = 1e-7
        delta = num / (den + eps)
        delta = torch.clamp(delta, -1.0, 1.0)

        return z_k - delta
