
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

# =============================================================================
# LOCAL LEARNING FUNCTIONS (Fixed: online traces, no storage of all timesteps)
# =============================================================================

class OSTL_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, cell, n_steps, decay):
        ctx.decay = decay
        ctx.n_steps = n_steps
        ctx.cell = cell
        
        device, dtype = z.device, z.dtype
        
        with torch.no_grad():
            h = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)
            
            for _ in range(n_steps):
                h = cell(h, z)
                h_trace = decay * h_trace + h
            
            sum_decay = (1 - decay**n_steps) / (1 - decay)
            z_trace = z * sum_decay
            
        ctx.save_for_backward(z, h_trace, z_trace)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        z, h_trace, z_trace = ctx.saved_tensors
        cell = ctx.cell
        
        with torch.no_grad():
            if cell.Wx.weight.grad is None:
                cell.Wx.weight.grad = torch.zeros_like(cell.Wx.weight)
            cell.Wx.weight.grad.add_(grad_output.t() @ z_trace)
            
            if cell.Wz.weight.grad is None:
                cell.Wz.weight.grad = torch.zeros_like(cell.Wz.weight)
            cell.Wz.weight.grad.add_(grad_output.t() @ h_trace)
            
            if cell.Wz.bias is not None:
                if cell.Wz.bias.grad is None:
                    cell.Wz.bias.grad = torch.zeros_like(cell.Wz.bias)
                cell.Wz.bias.grad.add_(grad_output.sum(0))

        return grad_output, None, None, None


class OSTTP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, cell, random_proj, n_steps, decay):
        ctx.decay = decay
        ctx.n_steps = n_steps
        ctx.cell = cell
        ctx.random_proj = random_proj
        
        device, dtype = z.device, z.dtype
        
        with torch.no_grad():
            h = torch.zeros_like(z)
            h_trace = torch.zeros_like(z)
            
            for _ in range(n_steps):
                h = cell(h, z)
                h_trace = decay * h_trace + h
                
            sum_decay = (1 - decay**n_steps) / (1 - decay)
            z_trace = z * sum_decay
            
        ctx.save_for_backward(z, h_trace, z_trace)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        z, h_trace, z_trace = ctx.saved_tensors
        cell = ctx.cell
        random_proj = ctx.random_proj
        
        projected_error = grad_output @ random_proj
        
        with torch.no_grad():
            if cell.Wx.weight.grad is None:
                cell.Wx.weight.grad = torch.zeros_like(cell.Wx.weight)
            cell.Wx.weight.grad.add_(projected_error.t() @ z_trace)
            
            if cell.Wz.weight.grad is None:
                cell.Wz.weight.grad = torch.zeros_like(cell.Wz.weight)
            cell.Wz.weight.grad.add_(projected_error.t() @ h_trace)
            
            if cell.Wz.bias is not None:
                if cell.Wz.bias.grad is None:
                    cell.Wz.bias.grad = torch.zeros_like(cell.Wz.bias)
                cell.Wz.bias.grad.add_(projected_error.sum(0))
            
        return grad_output, None, None, None, None


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class RecurrentCell(nn.Module):
    """Simplified cell for local learning (no LayerNorm)."""
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
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, backward_mode='phantom'):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.deqs = nn.ModuleList()
        for _ in range(num_layers):
            cell = RecurrentCell(hidden_dim)
            cfg = HybridConfig(max_iter=10, tol=1e-3, pjwr_iters=2, anderson_m=3)
            solver = SolverFactory.create(cfg, cell)
            self.deqs.append(DEQModule(cell, solver=solver, backward_mode=backward_mode))
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for deq in self.deqs:
            z = deq(z)
        return self.head(z)


class Deep_OSTL_Model(nn.Module):
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
            z = OSTL_Function.apply(z, cell, self.n_steps, 0.9)
        return self.head(z)


class Deep_OSTTP_Model(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, n_steps=8):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.cells = nn.ModuleList([RecurrentCell(hidden_dim) for _ in range(num_layers)])
        self.random_projections = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=False)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)
        self.n_steps = n_steps

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.proj_in(x)
        for i, cell in enumerate(self.cells):
            z = OSTTP_Function.apply(z, cell, self.random_projections[i], self.n_steps, 0.9)
        return self.head(z)


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
    # CIFAR-10: 32x32x3 = 3072 input features, 10 classes
    return trainloader, 3072, 10


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(layer_counts=[1, 3, 5, 8, 10, 12], epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- CIFAR-10 Layer Scaling Benchmark ({device}) ---")
    
    loader, in_dim, out_dim = get_cifar10_dataloader(batch_size=128)
    hidden_dim = 256
    
    results = {name: {} for name in ["Deep_BPTT", "Deep_Hybrid_DEQ", "Deep_OSTL", "Deep_OSTTP"]}
    
    for layers in layer_counts:
        print(f"\n=== Layers: {layers} ===")
        
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
            
            for ep in range(epochs):
                correct = total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    correct += (logits.argmax(dim=1) == y).sum().item()
                    total += y.size(0)
                acc = correct / total * 100
                if ep == epochs - 1:
                    final_acc = acc
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            elapsed = time.time() - start
            
            results[name][layers] = {"time": elapsed, "vram": peak_vram, "accuracy": final_acc}
            print(f"Time={elapsed:.1f}s VRAM={peak_vram:.0f}MB Acc={final_acc:.1f}%")
            
    return results, layer_counts


def plot(results, layers):
    os.makedirs("benchmark_plots", exist_ok=True)
    
    for metric, ylabel, fname in [
        ("vram", "VRAM (MB)", "layer_scaling_vram_cifar10.png"),
        ("time", "Time (seconds)", "layer_scaling_time_cifar10.png"),
        ("accuracy", "Accuracy (%)", "layer_scaling_accuracy_cifar10.png")
    ]:
        plt.figure(figsize=(8, 5))
        colors = {"Deep_BPTT": "blue", "Deep_Hybrid_DEQ": "green", 
                  "Deep_OSTL": "red", "Deep_OSTTP": "purple"}
        markers = {"Deep_BPTT": "o", "Deep_Hybrid_DEQ": "s", 
                   "Deep_OSTL": "^", "Deep_OSTTP": "d"}
        
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
    print("- layer_scaling_vram_cifar10.png")
    print("- layer_scaling_time_cifar10.png")
    print("- layer_scaling_accuracy_cifar10.png")


if __name__ == "__main__":
    layers = [1, 3, 5, 8, 10, 12]
    results, layers = run_benchmark(layer_counts=layers, epochs=5)
    plot(results, layers)
