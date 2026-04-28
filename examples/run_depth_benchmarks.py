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

class Deep_BPTT_Model(nn.Module):
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
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, solver_config=None, backward_mode='phantom'):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        
        self.deqs = nn.ModuleList()
        for _ in range(num_layers):
            cell = RecurrentCell(hidden_dim)
            if solver_config is None:
                # Use a fast hybrid solver for each layer
                layer_config = HybridConfig(max_iter=10, tol=1e-3, pjwr_iters=2, anderson_m=3)
            else:
                layer_config = solver_config
                
            solver = SolverFactory.create(layer_config, cell)
            deq = DEQModule(
                cell,
                solver=solver,
                backward_mode=backward_mode
            )
            self.deqs.append(deq)
            
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for deq in self.deqs:
            z = deq(z)
        return self.head(z)

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

def run_layer_scaling_benchmark(layer_counts=[1, 3, 5, 8, 10, 12], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Layer Scaling Benchmark ({device}) ---")
    
    loader, in_dim, out_dim = get_spiral_dataloader()
    hidden_dim = 64
    
    results = {"Deep_BPTT": {}, "Deep_Hybrid_DEQ": {}}
    
    for layers in layer_counts:
        print(f"\nBenchmarking Num Physical Layers = {layers}")
        
        configs = {
            "Deep_BPTT": Deep_BPTT_Model(layers, in_dim, hidden_dim, out_dim, n_steps=8),
            "Deep_Hybrid_DEQ": Deep_DEQ_Model(layers, in_dim, hidden_dim, out_dim, backward_mode='phantom'),
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
            
            results[name][layers] = {
                "time": end_time - start_time,
                "vram": peak_vram,
                "accuracy": accs[-1],
            }
            print(f"    Time={results[name][layers]['time']:.2f}s, VRAM={results[name][layers]['vram']:.2f}MB, Acc={results[name][layers]['accuracy']:.2f}%")
            
    return results, layer_counts

def main():
    layer_counts = [1, 3, 5, 8, 10, 12]
    results, layers = run_layer_scaling_benchmark(layer_counts=layer_counts, epochs=5)
    
    # Plotting
    os.makedirs("benchmark_plots", exist_ok=True)
    
    # Plot VRAM vs Layers
    plt.figure(figsize=(8, 5))
    bptt_vram = [results["Deep_BPTT"][l]["vram"] for l in layers]
    deq_vram = [results["Deep_Hybrid_DEQ"][l]["vram"] for l in layers]
    plt.plot(layers, bptt_vram, label="Deep BPTT", marker='o', color='blue', linewidth=2)
    plt.plot(layers, deq_vram, label="Deep Hybrid DEQ", marker='s', color='green', linewidth=2)
    plt.title("Peak VRAM Usage vs Number of Physical Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("VRAM (MB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_plots/layer_scaling_vram.png")
    plt.close()
    
    # Plot Time vs Layers
    plt.figure(figsize=(8, 5))
    bptt_time = [results["Deep_BPTT"][l]["time"] for l in layers]
    deq_time = [results["Deep_Hybrid_DEQ"][l]["time"] for l in layers]
    plt.plot(layers, bptt_time, label="Deep BPTT", marker='o', color='blue', linewidth=2)
    plt.plot(layers, deq_time, label="Deep Hybrid DEQ", marker='s', color='green', linewidth=2)
    plt.title("Training Time vs Number of Physical Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_plots/layer_scaling_time.png")
    plt.close()
    
    # Plot Accuracy vs Layers
    plt.figure(figsize=(8, 5))
    bptt_acc = [results["Deep_BPTT"][l]["accuracy"] for l in layers]
    deq_acc = [results["Deep_Hybrid_DEQ"][l]["accuracy"] for l in layers]
    plt.plot(layers, bptt_acc, label="Deep BPTT", marker='o', color='blue', linewidth=2)
    plt.plot(layers, deq_acc, label="Deep Hybrid DEQ", marker='s', color='green', linewidth=2)
    plt.title("Final Accuracy vs Number of Physical Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("benchmark_plots/layer_scaling_accuracy.png")
    plt.close()

    print("\nPlots saved to benchmark_plots/ directory:")
    print("- benchmark_plots/layer_scaling_vram.png")
    print("- benchmark_plots/layer_scaling_time.png")
    print("- benchmark_plots/layer_scaling_accuracy.png")

if __name__ == "__main__":
    main()
