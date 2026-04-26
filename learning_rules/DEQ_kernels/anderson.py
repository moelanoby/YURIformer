"""
Sketched Anderson Acceleration Solver.

Anderson Acceleration (Type-I) for fixed-point iteration, optimized with
Sketched Anderson Acceleration (SAA). Reduces solver linear algebra
overhead from O(m·D) to O(m·k) by projecting residuals onto a random
low-dimensional subspace.

Best for: Moderate-to-hard problems where you need robust global
convergence with low per-iteration cost.
"""

import torch


class AndersonSolver:
    """
    Sketched Anderson Acceleration for DEQ fixed-point solving.

    Parameters
    ----------
    f : callable
        The equilibrium map ``f(z, x)``.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Convergence tolerance on relative residual norm.
    m : int
        Anderson mixing window size (number of past iterates to store).
    beta : float
        Damping / mixing coefficient.
    sketch_size : int or None
        If not None and < D, uses random sketching to reduce the cost of
        the least-squares solve from O(m·D) to O(m·sketch_size).

    Example
    -------
    >>> solver = AndersonSolver(my_layer, max_iter=50, tol=1e-5, m=5)
    >>> z_star = solver.solve(x)
    """

    def __init__(self, f, max_iter=50, tol=1e-5, m=5, beta=1.0, sketch_size=256):
        self.f = f
        self.max_iter = max_iter
        self.tol = tol
        self.m = m
        self.beta = beta
        self.sketch_size = sketch_size

    def solve(self, x, z_init=None):
        """
        Run Anderson Acceleration to find the fixed point of ``f``.

        Parameters
        ----------
        x : Tensor
            Input injection.
        z_init : Tensor, optional
            Initial guess. Defaults to zeros.

        Returns
        -------
        Tensor
            Approximate fixed point z*.
        """
        z = z_init if z_init is not None else torch.zeros_like(x)
        orig_shape = z.shape
        bsz = z.shape[0]
        z_flat = z.reshape(bsz, -1)
        D = z_flat.shape[1]

        best_z = z_flat.clone()
        best_res = float('inf')
        rel_norm = float('inf')

        # Pre-allocate ring buffers
        dX = torch.zeros(bsz, D, self.m, device=z.device, dtype=z.dtype)
        dG = torch.zeros(bsz, D, self.m, device=z.device, dtype=z.dtype)

        # Sketching Setup
        use_sketch = self.sketch_size is not None and self.sketch_size < D
        if use_sketch:
            sketch_idx = torch.randint(0, D, (self.sketch_size,), device=z.device)

        g_k = self.f(z_flat.reshape(orig_shape), x).reshape(bsz, -1) - z_flat
        prev_z = z_flat.clone()
        prev_g = g_k.clone()

        z_flat = z_flat + self.beta * g_k
        valid_cols = 0
        ptr = 0

        for k in range(1, self.max_iter):
            g_k = self.f(z_flat.reshape(orig_shape), x).reshape(bsz, -1) - z_flat

            res_norm = g_k.norm(dim=-1).mean().item()
            rel_norm = res_norm / (z_flat.norm(dim=-1).mean().item() + 1e-9)

            if res_norm < best_res:
                best_res = res_norm
                best_z = z_flat.clone()

            if rel_norm < self.tol:
                break

            col_idx = ptr % self.m
            dX[:, :, col_idx] = z_flat - prev_z
            dG[:, :, col_idx] = g_k - prev_g

            prev_z = z_flat.clone()
            prev_g = g_k.clone()

            valid_cols = min(valid_cols + 1, self.m)
            ptr += 1

            dX_active = dX[:, :, :valid_cols]
            dG_active = dG[:, :, :valid_cols]

            if use_sketch:
                dG_sk = dG_active[:, sketch_idx, :]
                g_k_sk = g_k[:, sketch_idx]
                GtG = torch.bmm(dG_sk.transpose(1, 2), dG_sk)
                reg = 1e-6 * torch.eye(valid_cols, device=GtG.device, dtype=GtG.dtype).unsqueeze(0).expand(bsz, -1, -1)
                GtG = GtG + reg
                Gtg = torch.bmm(dG_sk.transpose(1, 2), g_k_sk.unsqueeze(-1))
            else:
                GtG = torch.bmm(dG_active.transpose(1, 2), dG_active)
                reg = 1e-6 * torch.eye(valid_cols, device=GtG.device, dtype=GtG.dtype).unsqueeze(0).expand(bsz, -1, -1)
                GtG = GtG + reg
                Gtg = torch.bmm(dG_active.transpose(1, 2), g_k.unsqueeze(-1))

            try:
                L = torch.linalg.cholesky(GtG)
                alpha = torch.cholesky_solve(Gtg, L).squeeze(-1)
            except RuntimeError:
                try:
                    alpha = torch.linalg.solve(GtG, Gtg).squeeze(-1)
                except RuntimeError:
                    fallback_dG = dG_sk if use_sketch else dG_active
                    fallback_gk = g_k_sk if use_sketch else g_k
                    alpha = torch.linalg.lstsq(fallback_dG, fallback_gk.unsqueeze(-1)).solution.squeeze(-1)

            z_flat_new = (
                (z_flat - torch.bmm(dX_active, alpha.unsqueeze(-1)).squeeze(-1))
                + self.beta * (g_k - torch.bmm(dG_active, alpha.unsqueeze(-1)).squeeze(-1))
            )

            if k > 2:
                g_new = self.f(z_flat_new.reshape(orig_shape), x).reshape(bsz, -1) - z_flat_new
                if g_new.norm(dim=-1).mean().item() > 3.0 * best_res:
                    z_flat = z_flat + self.beta * g_k
                    prev_z = z_flat.clone()
                else:
                    z_flat = z_flat_new
                    prev_z = z_flat.clone()
            else:
                z_flat = z_flat_new

        return best_z.reshape(orig_shape) if rel_norm >= self.tol else z_flat.reshape(orig_shape)
