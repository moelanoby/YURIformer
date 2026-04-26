"""
3-Phase Hybrid Solver: PJWR → Block-Parallel Anderson → Broyden.

Combines all three solvers into a pipeline that exploits each method's
strengths: PJWR for cheap parallel warm-up, Anderson for robust global
convergence, and Broyden for superlinear local refinement.
"""
import torch
import torch.nn.functional as F
from .accelerators import ShanksAccelerator


class HybridAndersonBroydenSolver:
    """
    3-Phase Hybrid DEQ Solver with Sketched Anderson Acceleration.

    Phase 1 (PJWR): Cheap parallel warm-up with Shanks acceleration.
    Phase 2 (Anderson): Block-parallel Anderson mixing for robust convergence.
    Phase 3 (Broyden): Quasi-Newton refinement for superlinear finish.

    Parameters
    ----------
    f : callable
        Equilibrium map ``f(z, x)``.
    max_iter : int
        Total iteration budget across all phases.
    tol : float
        Convergence tolerance.
    pjwr_iters : int
        Max iterations to spend in the PJWR warm-up phase.
    use_shanks : bool
        Use Shanks acceleration during PJWR phase.
    anderson_m : int
        Anderson mixing window size.
    anderson_beta : float
        Anderson damping coefficient.
    n_blocks : int
        Number of blocks for block-parallel Anderson step.
    sketch_size : int or None
        Sketch dimension for SAA.
    broyden_memory : int
        Broyden rank-1 update buffer size.
    switch_tol : float
        Residual threshold to switch from Anderson to Broyden.
    """
    def __init__(self, f, max_iter=50, tol=1e-5,
                 pjwr_iters=8, use_shanks=True,
                 anderson_m=5, anderson_beta=1.0, n_blocks=4, sketch_size=256,
                 broyden_memory=15, switch_tol=1e-2):
        self.f = f
        self.max_iter = max_iter
        self.tol = tol
        self.pjwr_iters = pjwr_iters
        self.use_shanks = use_shanks
        self.shanks = ShanksAccelerator()
        self.anderson_m = anderson_m
        self.anderson_beta = anderson_beta
        self.n_blocks = n_blocks
        self.sketch_size = sketch_size
        self.broyden_memory = broyden_memory
        self.switch_tol = switch_tol

    def _block_parallel_anderson_step(self, z_flat, g_k, dX_active, dG_active,
                                      valid_cols, bsz, D):
        n_blocks = min(self.n_blocks, D)
        remainder = D % n_blocks

        if remainder > 0:
            pad = n_blocks - remainder
            z_padded = F.pad(z_flat, (0, pad))
            g_padded = F.pad(g_k, (0, pad))
            dX_padded = F.pad(dX_active, (0, 0, 0, pad))
            dG_padded = F.pad(dG_active, (0, 0, 0, pad))
            D_padded = D + pad
        else:
            z_padded, g_padded = z_flat, g_k
            dX_padded, dG_padded = dX_active, dG_active
            D_padded = D

        block_size = D_padded // n_blocks
        BN = bsz * n_blocks

        z_merged = z_padded.reshape(BN, block_size)
        g_merged = g_padded.reshape(BN, block_size)
        dX_merged = dX_padded.reshape(bsz, n_blocks, block_size, valid_cols).reshape(BN, block_size, valid_cols)
        dG_merged = dG_padded.reshape(bsz, n_blocks, block_size, valid_cols).reshape(BN, block_size, valid_cols)

        use_sketch = self.sketch_size is not None and self.sketch_size < block_size
        if use_sketch:
            sketch_idx = torch.randint(0, block_size, (self.sketch_size,), device=z_flat.device)
            dG_sk = dG_merged[:, sketch_idx, :]
            g_sk = g_merged[:, sketch_idx]
            GtG = torch.bmm(dG_sk.transpose(1, 2), dG_sk)
            reg = 1e-6 * torch.eye(valid_cols, device=GtG.device, dtype=GtG.dtype).unsqueeze(0).expand(BN, -1, -1)
            GtG = GtG + reg
            Gtg = torch.bmm(dG_sk.transpose(1, 2), g_sk.unsqueeze(-1))
        else:
            GtG = torch.bmm(dG_merged.transpose(1, 2), dG_merged)
            reg = 1e-6 * torch.eye(valid_cols, device=GtG.device, dtype=GtG.dtype).unsqueeze(0).expand(BN, -1, -1)
            GtG = GtG + reg
            Gtg = torch.bmm(dG_merged.transpose(1, 2), g_merged.unsqueeze(-1))

        try:
            L = torch.linalg.cholesky(GtG)
            alpha = torch.cholesky_solve(Gtg, L).squeeze(-1)
        except RuntimeError:
            try:
                alpha = torch.linalg.solve(GtG, Gtg).squeeze(-1)
            except RuntimeError:
                fb_dG = dG_sk if use_sketch else dG_merged
                fb_gk = g_sk if use_sketch else g_merged
                alpha = torch.linalg.lstsq(fb_dG, fb_gk.unsqueeze(-1)).solution.squeeze(-1)

        z_new_merged = (
            (z_merged - torch.bmm(dX_merged, alpha.unsqueeze(-1)).squeeze(-1))
            + self.anderson_beta * (g_merged - torch.bmm(dG_merged, alpha.unsqueeze(-1)).squeeze(-1))
        )

        z_flat_new = z_new_merged.reshape(bsz, D_padded)
        if remainder > 0:
            z_flat_new = z_flat_new[:, :D]

        return z_flat_new

    def solve(self, x, z_init=None):
        z = z_init if z_init is not None else torch.zeros_like(x)
        orig_shape = z.shape
        bsz = z.shape[0]
        z_flat = z.reshape(bsz, -1)
        D = z_flat.shape[-1]

        def g(z_f):
            return self.f(z_f.reshape(orig_shape), x).reshape(bsz, -1) - z_f

        best_z = z_flat.clone()
        best_res = float('inf')
        rel_norm = float('inf')
        phase = 'pjwr'

        z_history = [z_flat.clone()]
        prev_pjwr_res = float('inf')
        pjwr_stall_count = 0

        dX_anderson = torch.zeros(bsz, D, self.anderson_m, device=z.device, dtype=z.dtype)
        dG_anderson = torch.zeros(bsz, D, self.anderson_m, device=z.device, dtype=z.dtype)
        anderson_valid = 0
        anderson_ptr = 0
        prev_z_aa = None
        prev_g_aa = None

        U_mat = torch.zeros(bsz, D, self.broyden_memory, device=z.device, dtype=z.dtype)
        VT_mat = torch.zeros(bsz, self.broyden_memory, D, device=z.device, dtype=z.dtype)
        broyden_valid = 0
        broyden_ptr = 0
        gx_broyden = None

        for k in range(self.max_iter):
            if phase == 'pjwr':
                f_z = self.f(z_flat.reshape(orig_shape), x).reshape(bsz, -1)
                g_k = f_z - z_flat
                z_flat = f_z

                res_norm = g_k.norm(dim=-1).mean().item()
                rel_norm = res_norm / (z_flat.norm(dim=-1).mean().item() + 1e-9)

                if res_norm < best_res:
                    best_res = res_norm
                    best_z = z_flat.clone()
                if rel_norm < self.tol:
                    break

                z_history.append(z_flat.clone())
                if self.use_shanks and len(z_history) >= 3:
                    z_flat = self.shanks.accelerate(z_history)
                    z_history[-1] = z_flat.clone()

                improvement = abs(prev_pjwr_res - res_norm) / (prev_pjwr_res + 1e-9)
                if improvement < 0.01:
                    pjwr_stall_count += 1
                else:
                    pjwr_stall_count = 0
                prev_pjwr_res = res_norm

                if k >= self.pjwr_iters or pjwr_stall_count >= 3:
                    phase = 'anderson'
                    anderson_valid = 0
                    anderson_ptr = 0
                    prev_z_aa = z_flat.clone()
                    prev_g_aa = g_k.clone()
                    z_flat = z_flat + self.anderson_beta * g_k

            elif phase == 'anderson':
                f_z = self.f(z_flat.reshape(orig_shape), x).reshape(bsz, -1)
                g_k = f_z - z_flat

                res_norm = g_k.norm(dim=-1).mean().item()
                rel_norm = res_norm / (z_flat.norm(dim=-1).mean().item() + 1e-9)

                if res_norm < best_res:
                    best_res = res_norm
                    best_z = z_flat.clone()
                if rel_norm < self.tol:
                    break

                if res_norm < self.switch_tol and anderson_valid >= 3:
                    phase = 'broyden'
                    gx_broyden = g_k.clone()
                    broyden_valid = 0
                    broyden_ptr = 0
                    continue

                col_idx = anderson_ptr % self.anderson_m
                dX_anderson[:, :, col_idx] = z_flat - prev_z_aa
                dG_anderson[:, :, col_idx] = g_k - prev_g_aa

                prev_z_aa = z_flat.clone()
                prev_g_aa = g_k.clone()

                anderson_valid = min(anderson_valid + 1, self.anderson_m)
                anderson_ptr += 1

                dX_active = dX_anderson[:, :, :anderson_valid]
                dG_active = dG_anderson[:, :, :anderson_valid]

                z_flat_new = self._block_parallel_anderson_step(
                    z_flat, g_k, dX_active, dG_active, anderson_valid, bsz, D
                )

                g_new = g(z_flat_new)
                if g_new.norm(dim=-1).mean().item() > 3.0 * best_res and anderson_valid > 2:
                    z_flat = z_flat + self.anderson_beta * g_k
                    prev_z_aa = z_flat.clone()
                else:
                    z_flat = z_flat_new

            elif phase == 'broyden':
                if broyden_valid == 0:
                    dz = -gx_broyden
                else:
                    U_act = U_mat[:, :, :broyden_valid]
                    VT_act = VT_mat[:, :broyden_valid, :]
                    inner = torch.bmm(VT_act, gx_broyden.unsqueeze(-1))
                    dz = -gx_broyden + torch.bmm(U_act, inner).squeeze(-1)

                z_flat_new = z_flat + dz
                gx_new = g(z_flat_new)

                res_norm = gx_new.norm(dim=-1).mean().item()
                rel_norm = res_norm / (z_flat_new.norm(dim=-1).mean().item() + 1e-9)

                if res_norm < best_res:
                    best_res = res_norm
                    best_z = z_flat_new.clone()
                if rel_norm < self.tol:
                    z_flat = z_flat_new
                    break

                if res_norm > 10.0 * best_res:
                    z_flat = best_z.clone()
                    gx_broyden = g(z_flat)
                    broyden_valid = 0
                    broyden_ptr = 0
                    continue

                s = dz
                dg_vec = gx_new - gx_broyden

                if broyden_valid == 0:
                    Jinv_dg = -dg_vec
                    vt = -s.clone()
                else:
                    U_act = U_mat[:, :, :broyden_valid]
                    VT_act = VT_mat[:, :broyden_valid, :]
                    inner_dg = torch.bmm(VT_act, dg_vec.unsqueeze(-1))
                    Jinv_dg = -dg_vec + torch.bmm(U_act, inner_dg).squeeze(-1)
                    inner_s = torch.bmm(VT_act, s.unsqueeze(-1))
                    vt = -s + torch.bmm(U_act, inner_s).squeeze(-1)

                numerator = s - Jinv_dg
                denominator = (vt * dg_vec).sum(dim=-1, keepdim=True) + 1e-10
                u = numerator / denominator

                col_idx = broyden_ptr % self.broyden_memory
                U_mat[:, :, col_idx] = u
                VT_mat[:, col_idx, :] = vt

                broyden_valid = min(broyden_valid + 1, self.broyden_memory)
                broyden_ptr += 1

                z_flat = z_flat_new
                gx_broyden = gx_new

        final_z = z_flat if rel_norm < self.tol else best_z
        return final_z.reshape(orig_shape)
