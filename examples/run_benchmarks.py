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

# Setup data loaders
def get_dataloaders(dataset_name, batch_size=128):
    if dataset_name == "Spiral":
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
    elif dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True), 28*28, 10
    elif dataset_name == "CIFAR-10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True), 3*32*32, 10
    else:
        raise ValueError("Unknown dataset")

def run_benchmark(dataset_name, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running benchmark on {dataset_name} ({device}) ---")
    
    loader, in_dim, out_dim = get_dataloaders(dataset_name)
    hidden_dim = 64
    
    configs = {
        "BPTT": BPTT_Model(in_dim, hidden_dim, out_dim, n_steps=10),
        "PJWR": DEQ_Configurable(in_dim, hidden_dim, out_dim, solver_config=PJWRConfig(max_iter=15, tol=1e-3)),
        "Anderson": DEQ_Configurable(in_dim, hidden_dim, out_dim, solver_config=AndersonConfig(max_iter=15, tol=1e-3, m=5)),
        "Broyden": DEQ_Configurable(in_dim, hidden_dim, out_dim, solver_config=BroydenConfig(max_iter=15, tol=1e-3, memory=5)),
        "Hybrid": DEQ_Configurable(in_dim, hidden_dim, out_dim, solver_config=HybridConfig(max_iter=15, tol=1e-3, pjwr_iters=3, anderson_m=4)),
    }
    
    results = {}
    
    for name, model in configs.items():
        print(f"Training {name}...")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        start_time = time.time()
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
        
        results[name] = {
            "time": end_time - start_time,
            "vram": peak_vram,
            "accuracy": accs[-1],
            "acc_history": accs
        }
        print(f"  {name}: Time={results[name]['time']:.2f}s, VRAM={results[name]['vram']:.2f}MB, Acc={results[name]['accuracy']:.2f}%")
        
    return results

def main():
    datasets = ["Spiral", "MNIST", "CIFAR-10"]
    all_results = {}
    
    for ds in datasets:
        # Fewer epochs for larger datasets to save time during this run, but enough to see trends
        eps = 15 if ds == "Spiral" else 3
        all_results[ds] = run_benchmark(ds, epochs=eps)
        
    # Plotting
    os.makedirs("benchmark_plots", exist_ok=True)
    
    for ds in datasets:
        res = all_results[ds]
        names = list(res.keys())
        
        # Plot Accuracy History
        plt.figure(figsize=(8, 5))
        for name in names:
            plt.plot(res[name]["acc_history"], label=name, marker='o')
        plt.title(f"{ds} - Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"benchmark_plots/{ds}_accuracy.png")
        plt.close()
        
        # Plot VRAM
        plt.figure(figsize=(8, 5))
        vrams = [res[n]["vram"] for n in names]
        plt.bar(names, vrams, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.title(f"{ds} - Peak VRAM Usage")
        plt.ylabel("VRAM (MB)")
        for i, v in enumerate(vrams):
            plt.text(i, v, f"{v:.1f}", ha='center', va='bottom')
        plt.grid(axis='y')
        plt.savefig(f"benchmark_plots/{ds}_vram.png")
        plt.close()
        
        # Plot Time
        plt.figure(figsize=(8, 5))
        times = [res[n]["time"] for n in names]
        plt.bar(names, times, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.title(f"{ds} - Total Training Time")
        plt.ylabel("Time (seconds)")
        for i, t in enumerate(times):
            plt.text(i, t, f"{t:.1f}", ha='center', va='bottom')
        plt.grid(axis='y')
        plt.savefig(f"benchmark_plots/{ds}_time.png")
        plt.close()

if __name__ == "__main__":
    main()
