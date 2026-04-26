"""
DEQ as a Drop-in Replacement for BPTT
======================================

This example demonstrates how to convert ANY recurrent / weight-tied
model from BPTT training to DEQ equilibrium training with minimal code
changes.  We show three progressively advanced patterns:

  1. Minimal swap  — 4 lines of code changed
  2. Solver picker — choose Anderson / Broyden / PJWR / Hybrid via config
  3. Full pipeline — Jacobian regularization + backward mode comparison

Run:
    python examples/deq_dropin_replace_bptt.py
"""

import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Make the DEQ_kernels package importable ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning_rules"))

from DEQ_kernels import (
    # Solvers
    ParallelJacobiWaveformSolver,
    AndersonSolver,
    BroydenSolver,
    HybridAndersonBroydenSolver,
    # Drop-in module
    DEQModule,
    # Config-based construction
    PJWRConfig, AndersonConfig, BroydenConfig, HybridConfig,
    SolverFactory,
    # Regularization
    jacobian_spectral_norm,
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 0: Define a recurrent cell (shared by BOTH BPTT and DEQ)
# ═════════════════════════════════════════════════════════════════════════════

class RecurrentCell(nn.Module):
    """
    A simple weight-tied recurrent cell:
        z_{k+1} = tanh(W_z @ z_k + W_x @ x + b)

    This is the *only* module you write.  BPTT unrolls it N times;
    DEQ finds its fixed point z* = f(z*, x) implicitly.
    """
    def __init__(self, dim):
        super().__init__()
        self.Wz = nn.Linear(dim, dim)
        self.Wx = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        # Spectral-norm-friendly init (small residual around identity)
        nn.init.eye_(self.Wz.weight)
        self.Wz.weight.data *= 0.5
        nn.init.xavier_uniform_(self.Wx.weight, gain=0.8)

    def forward(self, z, x):
        return torch.tanh(self.norm(self.Wz(z) + self.Wx(x)))


# ═════════════════════════════════════════════════════════════════════════════
# PATTERN 1 — BPTT Baseline (the "before" code)
# ═════════════════════════════════════════════════════════════════════════════

class BPTT_Model(nn.Module):
    """Standard BPTT: unroll the cell for a fixed number of steps."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_steps=20):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cell = RecurrentCell(hidden_dim)       # <-- same cell
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps

    def forward(self, x):
        x = self.proj_in(x)
        z = torch.zeros_like(x)
        for _ in range(self.n_steps):               # <-- BPTT unrolling
            z = self.cell(z, x)
        return self.head(z)


# ═════════════════════════════════════════════════════════════════════════════
# PATTERN 2 — Minimal DEQ swap  (the "after" code — 4 lines changed)
# ═════════════════════════════════════════════════════════════════════════════

class DEQ_Model(nn.Module):
    """
    DEQ drop-in: replace the for-loop with DEQModule.
    
    Changed lines vs BPTT_Model:
      - self.deq = DEQModule(self.cell)            # wraps the cell
      - z_star = self.deq(x)                       # replaces the loop
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cell = RecurrentCell(hidden_dim)       # <-- SAME cell
        self.deq = DEQModule(self.cell)             # ← NEW: wrap with DEQ
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        z_star = self.deq(x)                        # ← NEW: replaces for-loop
        return self.head(z_star)


# ═════════════════════════════════════════════════════════════════════════════
# PATTERN 3 — Choose your solver via Config
# ═════════════════════════════════════════════════════════════════════════════

class DEQ_Configurable(nn.Module):
    """
    Pick any solver at construction time using the config system.

    Example configs:
        PJWRConfig(max_iter=40, tol=1e-3)
        AndersonConfig(max_iter=30, m=8, sketch_size=128)
        BroydenConfig(max_iter=30, memory=15)
        HybridConfig(max_iter=40, pjwr_iters=5, anderson_m=6)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, solver_config=None,
                 backward_mode='phantom'):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cell = RecurrentCell(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

        # Build solver from config (or default to Hybrid)
        if solver_config is None:
            solver_config = HybridConfig()
        solver = SolverFactory.create(solver_config, self.cell)

        self.deq = DEQModule(
            self.cell,
            solver=solver,
            backward_mode=backward_mode,  # 'phantom' | 'neumann-1' | 'ift'
        )

    def forward(self, x):
        x = self.proj_in(x)
        z_star = self.deq(x)
        return self.head(z_star)


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def make_spiral_dataset(n=800, noise=0.25):
    """2-class spiral for quick benchmarking."""
    t = torch.linspace(0, 3 * math.pi, n // 2)
    r = t / (3 * math.pi)
    X = torch.cat([
        torch.stack([r * torch.cos(t), r * torch.sin(t)], 1),
        torch.stack([r * torch.cos(t + math.pi), r * torch.sin(t + math.pi)], 1),
    ]) + noise * torch.randn(n, 2)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    return X, y


def train_and_eval(model, X, y, name, epochs=200, lr=2e-3, jac_reg=False):
    """Train a model and print accuracy + wall time."""
    opt = optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()

    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        # Optional: Jacobian regularization for DEQ stability
        if jac_reg and hasattr(model, 'cell') and hasattr(model, 'deq'):
            x_proj = model.proj_in(X)
            with torch.no_grad():
                z_star = model.deq(x_proj)
            jac_pen = jacobian_spectral_norm(model.cell, z_star, x_proj,
                                             n_power_iters=3, target=0.9)
            loss = loss + 0.01 * jac_pen

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    elapsed = time.time() - t0
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item() * 100

    print(f"  {name:<45s}  acc={acc:5.1f}%  time={elapsed:.2f}s")
    return acc


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Run all patterns side by side
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    X, y = make_spiral_dataset()

    IN, HID, OUT = 2, 64, 2

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   DEQ Solvers — Drop-in BPTT Replacement Benchmark         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── 1. BPTT Baseline ────────────────────────────────────────────────
    print("\n── Pattern 1: BPTT Baseline ──")
    for steps in [5, 20]:
        train_and_eval(
            BPTT_Model(IN, HID, OUT, n_steps=steps), X, y,
            f"BPTT (unroll={steps})"
        )

    # ── 2. Minimal DEQ swap ─────────────────────────────────────────────
    print("\n── Pattern 2: Minimal DEQ Drop-in (default Hybrid solver) ──")
    train_and_eval(
        DEQ_Model(IN, HID, OUT), X, y,
        "DEQ (Hybrid, phantom grad)"
    )

    # ── 3. Per-solver comparison ────────────────────────────────────────
    print("\n── Pattern 3: Pick Your Solver via Config ──")

    solver_configs = {
        "PJWR (parallel warm-up only)":
            PJWRConfig(max_iter=40, tol=1e-3),

        "Anderson (sketched, m=8)":
            AndersonConfig(max_iter=40, tol=1e-4, m=8, sketch_size=128),

        "Broyden (quasi-Newton, mem=15)":
            BroydenConfig(max_iter=40, tol=1e-4, memory=15),

        "Hybrid (PJWR→Anderson→Broyden)":
            HybridConfig(max_iter=40, tol=1e-4, pjwr_iters=5,
                         anderson_m=6, broyden_memory=10),
    }

    for label, cfg in solver_configs.items():
        train_and_eval(
            DEQ_Configurable(IN, HID, OUT, solver_config=cfg), X, y,
            label,
        )

    # ── 4. Backward mode comparison ─────────────────────────────────────
    print("\n── Pattern 4: Backward Mode Comparison ──")
    for mode in ['phantom', 'neumann-1']:
        train_and_eval(
            DEQ_Configurable(IN, HID, OUT,
                             solver_config=AndersonConfig(max_iter=30),
                             backward_mode=mode),
            X, y,
            f"Anderson + backward={mode}",
        )

    # ── 5. With Jacobian regularization ─────────────────────────────────
    print("\n── Pattern 5: Jacobian Regularization for Stability ──")
    train_and_eval(
        DEQ_Model(IN, HID, OUT), X, y,
        "DEQ Hybrid + Jacobian Reg (λ=0.01)",
        jac_reg=True,
    )

    print("\n✅ All benchmarks complete.")
    print("   → BPTT uses O(N) memory and O(N) backward time.")
    print("   → DEQ uses O(1) memory and O(1) backward with phantom grads.")
    print("   → Same cell, same weights — just swap the training loop.")
