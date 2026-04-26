"""
Limited-Memory Broyden's Method Solver.
"""
import torch


class BroydenSolver:
    """
    Limited-Memory Broyden's Method for DEQ fixed-point solving.
    Uses batched tensor ops and circular buffers — no Python loops over history.
    """
    def __init__(self, f, max_iter=50, tol=1e-5, memory=20):
        self.f = f
        self.max_iter = max_iter
        self.tol = tol
        self.memory = memory

    def solve(self, x, z_init=None):
        z = z_init if z_init is not None else torch.zeros_like(x)
        orig_shape = z.shape
        bsz = z.shape[0]
        z_flat = z.reshape(bsz, -1)
        D = z_flat.shape[1]

        def g(z_f):
            return self.f(z_f.reshape(orig_shape), x).reshape(bsz, -1) - z_f

        gx = g(z_flat)
        best_z = z_flat.clone()
        best_res = gx.norm(dim=-1).mean().item()

        U_mat = torch.zeros(bsz, D, self.memory, device=z.device, dtype=z.dtype)
        VT_mat = torch.zeros(bsz, self.memory, D, device=z.device, dtype=z.dtype)
        valid_cols = 0
        ptr = 0
        rel_norm = float('inf')

        for k in range(self.max_iter):
            if valid_cols == 0:
                dz = -gx
            else:
                U_active = U_mat[:, :, :valid_cols]
                VT_active = VT_mat[:, :valid_cols, :]
                inner = torch.bmm(VT_active, gx.unsqueeze(-1))
                dz = -gx + torch.bmm(U_active, inner).squeeze(-1)

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
                gx = g(z_flat)
                valid_cols = 0
                ptr = 0
                continue

            s = dz
            dg = gx_new - gx

            if valid_cols == 0:
                Jinv_dg = -dg
                vt = -s.clone()
            else:
                U_active = U_mat[:, :, :valid_cols]
                VT_active = VT_mat[:, :valid_cols, :]
                inner_dg = torch.bmm(VT_active, dg.unsqueeze(-1))
                Jinv_dg = -dg + torch.bmm(U_active, inner_dg).squeeze(-1)
                inner_s = torch.bmm(VT_active, s.unsqueeze(-1))
                vt = -s + torch.bmm(U_active, inner_s).squeeze(-1)

            numerator = s - Jinv_dg
            denominator = (vt * dg).sum(dim=-1, keepdim=True) + 1e-10
            u = numerator / denominator

            col_idx = ptr % self.memory
            U_mat[:, :, col_idx] = u
            VT_mat[:, col_idx, :] = vt

            valid_cols = min(valid_cols + 1, self.memory)
            ptr += 1

            z_flat = z_flat_new
            gx = gx_new

        return best_z.reshape(orig_shape) if rel_norm >= self.tol else z_flat.reshape(orig_shape)
