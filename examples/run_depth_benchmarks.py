"""
YURIformer Benchmark: CIFAR-10
Fixed:
  - DEQ: single shared cell, ONE equilibrium (weight-tied, standard DEQ)
  - OSTL/OSTTP: online traces + manual training loop (no autograd.Function, no save_for_backward)
"""
import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =============================================================================
# PATH SETUP
# =============================================================================
# Add the repo root so 'learning_rules' is importable as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from learning_rules.DEQ_kernels import (
    DEQModule, HybridConfig, SolverFactory
)

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class RecurrentCell(nn.Module):
    """No LayerNorm — all params must be manually updatable for local learning."""
    def __init__(self, dim):
        super().__init__()
        self.Wz = nn.Linear(dim, dim)
        self.Wx = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.Wz.weight)
        self.Wz.weight.data *= 0.5
        nn.init.xavier_uniform_(self.Wx.weight, gain=0.8)

    def forward(self, z, x):
        return torch.tanh(self.Wz(z) + self.Wx(x))


class Deep_BPTT_Model(nn.Module):
    """Standard BPTT with num_layers DIFFERENT cells."""
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, n_steps=8):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cells = nn.ModuleList([RecurrentCell(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for cell in self.cells:
            h = torch.zeros_like(z)
            for _ in range(self.n_steps):
                h = cell(h, z)
            z = h
        return self.head(z)


class Deep_DEQ_Model(nn.Module):
    """
    FIXED: Single shared cell, ONE equilibrium solve.
    Standard weight-tied DEQ: the same cell is iterated to convergence.
    num_layers is ignored — depth is implicit in the fixed-point solve.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, backward_mode='phantom'):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        # ONE shared cell for the entire depth
        self.cell = RecurrentCell(hidden_dim)
        cfg = HybridConfig(max_iter=10, tol=1e-3, pjwr_iters=2, anderson_m=3)
        solver = SolverFactory.create(cfg, self.cell)
        self.deq = DEQModule(self.cell, solver=solver, backward_mode=backward_mode)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        z = self.deq(z)  # ONE equilibrium = infinite implicit depth
        return self.head(z)


class Deep_OSTL_Model(nn.Module):
    """
    Online local learning with eligibility traces.
    Use local_forward() + manual_train_step() instead of loss.backward().
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, n_steps=8, decay=0.9):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cells = nn.ModuleList([RecurrentCell(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps
        self.decay = decay

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for cell in self.cells:
            h = torch.zeros_like(z)
            for _ in range(self.n_steps):
                h = cell(h, z)
            z = h
        return self.head(z)

    def local_forward(self, x):
        """Returns logits + traces for manual local learning."""
        x_flat = x.view(x.size(0), -1)
        z = self.proj_in(x_flat)
        traces = []  # (z_trace, h_trace, cell) per layer
        for cell in self.cells:
            h = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)
            for _ in range(self.n_steps):
                h = cell(h, z)
                h_trace = self.decay * h_trace + h
            sum_decay = (1 - self.decay**self.n_steps) / (1 - self.decay)
            z_trace = z * sum_decay
            traces.append((z_trace, h_trace, cell))
            z = h
        logits = self.head(z)
        return logits, traces, x_flat, z


class Deep_OSTTP_Model(nn.Module):
    """
    Online local learning with Random Target Projection.
    Use local_forward() + manual_train_step() instead of loss.backward().
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, n_steps=8, decay=0.9):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cells = nn.ModuleList([RecurrentCell(hidden_dim) for _ in range(num_layers)])
        self.random_projections = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=False)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps
        self.decay = decay

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for i, cell in enumerate(self.cells):
            h = torch.zeros_like(z)
            for _ in range(self.n_steps):
                h = cell(h, z)
            z = h
        return self.head(z)

    def local_forward(self, x):
        """Returns logits + traces for manual local learning."""
        x_flat = x.view(x.size(0), -1)
        z = self.proj_in(x_flat)
        traces = []
        for i, cell in enumerate(self.cells):
            h = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)
            for _ in range(self.n_steps):
                h = cell(h, z)
                h_trace = self.decay * h_trace + h
            sum_decay = (1 - self.decay**self.n_steps) / (1 - self.decay)
            z_trace = z * sum_decay
            traces.append((z_trace, h_trace, cell, self.random_projections[i]))
            z = h
        logits = self.head(z)
        return logits, traces, x_flat, z


# =============================================================================
# CIFAR-10 DATA
# =============================================================================

def get_cifar10_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader, 3072, 10


# =============================================================================
# TRAINING HELPERS
# =============================================================================
from learning_rules.neuromorphic_kernels import manual_train_step_ostl, manual_train_step_osttp


def standard_train_step(model, x, y, optimizer):
    """Standard backprop for BPTT and DEQ."""
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()



# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(layer_counts=[1, 3, 5, 8, 10, 12], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== CIFAR-10 Layer Scaling Benchmark ({device}) ===")
    print("FIXED: DEQ = 1 equilibrium (weight-tied). OSTL/OSTTP = online updates (no autograd.Function).\n")

    loader, in_dim, out_dim = get_cifar10_dataloader(batch_size=128)
    hidden_dim = 256

    results = {name: {} for name in ["Deep_BPTT", "Deep_Hybrid_DEQ", "Deep_OSTL", "Deep_OSTTP"]}

    for layers in layer_counts:
        print(f"--- Layers: {layers} ---")

        configs = {
            "Deep_BPTT": Deep_BPTT_Model(layers, in_dim, hidden_dim, out_dim, n_steps=8),
            "Deep_Hybrid_DEQ": Deep_DEQ_Model(layers, in_dim, hidden_dim, out_dim),
            "Deep_OSTL": Deep_OSTL_Model(layers, in_dim, hidden_dim, out_dim, n_steps=8),
            "Deep_OSTTP": Deep_OSTTP_Model(layers, in_dim, hidden_dim, out_dim, n_steps=8),
        }

        for name, model in configs.items():
            print(f"  {name}...", end=" ", flush=True)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            final_acc = 0.0
            for ep in range(epochs):
                correct = total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    if name in ("Deep_OSTL", "Deep_OSTTP"):
                        if name == "Deep_OSTL":
                            manual_train_step_ostl(model, x, y, optimizer)
                        else:
                            manual_train_step_osttp(model, x, y, optimizer)
                        # Compute accuracy
                        with torch.no_grad():
                            logits = model(x)
                    else:
                        standard_train_step(model, x, y, optimizer)
                        with torch.no_grad():
                            logits = model(x)

                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

                acc = correct / total * 100
                final_acc = acc

            peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            elapsed = time.time() - start

            results[name][layers] = {"time": elapsed, "vram": peak_vram, "accuracy": final_acc}
            print(f"Time={elapsed:.1f}s VRAM={peak_vram:.0f}MB Acc={final_acc:.1f}%")

    return results, layer_counts


def plot(results, layers):
    os.makedirs("benchmark_plots", exist_ok=True)

    for metric, ylabel, fname in [
        ("vram", "VRAM (MB)", "layer_scaling_vram_cifar10_fixed.png"),
        ("time", "Time (seconds)", "layer_scaling_time_cifar10_fixed.png"),
        ("accuracy", "Accuracy (%)", "layer_scaling_accuracy_cifar10_fixed.png")
    ]:
        plt.figure(figsize=(8, 5))
        colors = {
            "Deep_BPTT": "blue", "Deep_Hybrid_DEQ": "green",
            "Deep_OSTL": "red", "Deep_OSTTP": "purple"
        }
        markers = {
            "Deep_BPTT": "o", "Deep_Hybrid_DEQ": "s",
            "Deep_OSTL": "^", "Deep_OSTTP": "d"
        }

        for name in colors:
            vals = [results[name][l][metric] for l in layers]
            plt.plot(layers, vals, label=name.replace("_", " "),
                     marker=markers[name], color=colors[name], linewidth=2, markersize=7)

        plt.title(f"{ylabel.split(' ')[0]} vs Number of Physical Layers (CIFAR-10)")
        plt.xlabel("Number of Layers")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"benchmark_plots/{fname}")
        plt.close()

    print("\nPlots saved to benchmark_plots/")
    print("- layer_scaling_vram_cifar10_fixed.png")
    print("- layer_scaling_time_cifar10_fixed.png")
    print("- layer_scaling_accuracy_cifar10_fixed.png")


if __name__ == "__main__":
    layers = [1, 3, 5, 8, 10, 12]
    results, layers = run_benchmark(layer_counts=layers, epochs=5)
    plot(results, layers)
