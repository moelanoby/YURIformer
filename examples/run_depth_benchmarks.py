import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Make DEQ_kernels importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning_rules"))

from DEQ_kernels import (
    DEQModule,
    PJWRConfig, AndersonConfig, BroydenConfig, HybridConfig,
    SolverFactory
)

class RecurrentCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wz = nn.Linear(dim, dim)
        self.Wx = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        nn.init.eye_(self.Wz.weight)
        self.Wz.weight.data *= 0.5
        nn.init.xavier_uniform_(self.Wx.weight, gain=0.8)

    def forward(self, z, x):
        return torch.tanh(self.norm(self.Wz(z) + self.Wx(x)))

class BPTT_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_steps=10):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cell = RecurrentCell(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.proj_in(x)
        z = torch.zeros_like(x)
        for _ in range(self.n_steps):
            z = self.cell(z, x)
        return self.head(z)

class DEQ_Configurable(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, solver_config=None, backward_mode='phantom'):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cell = RecurrentCell(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
        if solver_config is None:
            solver_config = HybridConfig()
        solver = SolverFactory.create(solver_config, self.cell)
        self.deq = DEQModule(
            self.cell,
            solver=solver,
            backward_mode=backward_mode
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.proj_in(x)
        z_star = self.deq(x)
        return self.head(z_star)

# Setup data loaders (using only Spiral for fast depth benchmarking)
def get_spiral_dataloader(batch_size=128):
    n = 2000
    t = torch.linspace(0, 3 * math.pi, n // 2)
    r = t / (3 * math.pi)
    X = torch.cat([
        torch.stack([r * torch.cos(t), r * torch.sin(t)], 1),
        torch.stack([r * torch.cos(t + math.pi), r * torch.sin(t + math.pi)], 1),
    ]) + 0.25 * torch.randn(n, 2)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), 2, 2

def run_depth_benchmark(depths=[5, 10, 12, 15, 20], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Depth Scaling Benchmark ({device}) ---")
    
    loader, in_dim, out_dim = get_spiral_dataloader()
    hidden_dim = 64
    
    results = {"BPTT": {}, "Hybrid_DEQ": {}}
    
    for depth in depths:
        print(f"\nBenchmarking Depth / Max Iterations = {depth}")
        
        configs = {
            "BPTT": BPTT_Model(in_dim, hidden_dim, out_dim, n_steps=depth),
            "Hybrid_DEQ": DEQ_Configurable(in_dim, hidden_dim, out_dim, solver_config=HybridConfig(max_iter=depth, tol=1e-4, pjwr_iters=3, anderson_m=4)),
        }
        
        for name, model in configs.items():
            print(f"  Training {name}...")
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            accs = []
            for ep in range(epochs):
                correct = 0
                total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                
                acc = correct / total * 100
                accs.append(acc)
                
            end_time = time.time()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            results[name][depth] = {
                "time": end_time - start_time,
                "vram": peak_vram,
                "accuracy": accs[-1],
            }
            print(f"    Time={results[name][depth]['time']:.2f}s, VRAM={results[name][depth]['vram']:.2f}MB, Acc={results[name][depth]['accuracy']:.2f}%")
            
    return results, depths

def main():
    depths_to_test = [5, 8, 10, 12, 15, 20]
    results, depths = run_depth_benchmark(depths=depths_to_test, epochs=5)
    
    # Plotting
    os.makedirs("benchmark_plots", exist_ok=True)
    
    # Plot VRAM vs Depth
    plt.figure(figsize=(8, 5))
    bptt_vram = [results["BPTT"][d]["vram"] for d in depths]
    deq_vram = [results["Hybrid_DEQ"][d]["vram"] for d in depths]
    plt.plot(depths, bptt_vram, label="BPTT", marker='o', color='blue', linewidth=2)
    plt.plot(depths, deq_vram, label="Hybrid DEQ (O(1) VRAM)", marker='s', color='green', linewidth=2)
    plt.title("Peak VRAM Usage vs Sequence Depth / Iterations")
    plt.xlabel("Depth (n_steps for BPTT, max_iter for DEQ)")
    plt.ylabel("VRAM (MB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_plots/depth_scaling_vram.png")
    plt.close()
    
    # Plot Time vs Depth
    plt.figure(figsize=(8, 5))
    bptt_time = [results["BPTT"][d]["time"] for d in depths]
    deq_time = [results["Hybrid_DEQ"][d]["time"] for d in depths]
    plt.plot(depths, bptt_time, label="BPTT", marker='o', color='blue', linewidth=2)
    plt.plot(depths, deq_time, label="Hybrid DEQ", marker='s', color='green', linewidth=2)
    plt.title("Training Time vs Sequence Depth / Iterations")
    plt.xlabel("Depth (n_steps for BPTT, max_iter for DEQ)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_plots/depth_scaling_time.png")
    plt.close()

    print("\nPlots saved to benchmark_plots/ directory:")
    print("- benchmark_plots/depth_scaling_vram.png")
    print("- benchmark_plots/depth_scaling_time.png")

if __name__ == "__main__":
    main()
