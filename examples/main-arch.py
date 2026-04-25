"""
Hierarchical Dendritic Computing Tree — True Branch Edition
===========================================================
Based on:
  - Multiplicative Coincidence Detection (Hadamard Dendrites)
  - Hierarchical Branching Structures
  - Dendritic Spike Non-linearities (Tanh/Softplus)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    HAS_TORCHVISION = True
except Exception as e:
    HAS_TORCHVISION = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

# Add paths for local library imports
sys.path.append(os.path.join(os.getcwd(), "DEQ_kernel"))
sys.path.append(os.path.join(os.getcwd(), "architecture_kernels"))

try:
    from deq_solver import DEQModule, jacobian_spectral_norm
except ImportError:
    # Fallback if pathing is tricky in the current environment
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../DEQ_kernel")
    from deq_solver import DEQModule, jacobian_spectral_norm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FAST APPROXIMATIONS (PWL for Exp and Log)
# ─────────────────────────────────────────────────────────────────────────────

class DendriticBranchNode(nn.Module):
    """ 
    True Hierarchical Dendritic Branch Node.
    Implements non-linear coincidence detection via multiplicative integration,
    modeling how biological dendrites integrate synaptic inputs.
    """
    def __init__(self, n_units):
        super().__init__()
        # Synaptic gains for the converging branches
        self.g_left = nn.Parameter(torch.randn(1, n_units) * 0.02 + 1.0)
        self.g_right = nn.Parameter(torch.randn(1, n_units) * 0.02 + 1.0)
        self.bias = nn.Parameter(torch.zeros(1, n_units))
        
    def forward(self, left, right):
        # Multiplicative interaction simulates coincidence detection (Logical AND)
        # We use tanh to model the saturating 'dendritic spike' behavior
        return torch.tanh((self.g_left * left) * (self.g_right * right) + self.bias)

class DendriticNeuron(nn.Module):
    """ Compatibility wrapper for a single Dendritic neuron """
    def __init__(self, in_features, depth=2):
        super().__init__()
        self.tree = DendriticTree(in_features, 1, depth)
        self.soma_gain = nn.Parameter(torch.ones(1))
        self.soma_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        root, levels = self.tree(x)
        soma = F.leaky_relu(self.soma_gain * root.squeeze(-1) + self.soma_bias, 0.1)
        return soma, levels


class DendriticTree(nn.Module):
    """ Vectorized Dendritic Tree for a population of neurons """
    def __init__(self, in_features, n_neurons, depth=2):
        super().__init__()
        self.depth = depth
        self.n_neurons = n_neurons
        self.n_leaves = 2 ** depth

        # Redundant mapping indices: (n_neurons, n_leaves, leaf_size)
        self.leaf_size = min(in_features, 8)
        indices = torch.stack([torch.stack([torch.randperm(in_features)[:self.leaf_size] 
                                          for _ in range(self.n_leaves)]) 
                               for _ in range(n_neurons)])
        self.register_buffer('indices', indices)

        # Vectorized Leaf Weights: (n_neurons, n_leaves, leaf_size)
        self.leaf_w = nn.Parameter(torch.randn(n_neurons, self.n_leaves, self.leaf_size) * 0.1)
        self.leaf_b = nn.Parameter(torch.zeros(n_neurons, self.n_leaves))

        # Vectorized Dendritic Nodes for each tree level
        self.tree_levels = nn.ModuleList()
        curr_branches = self.n_leaves
        for _ in range(depth):
            curr_branches //= 2
            # One node-group per level (handling all neurons in parallel)
            self.tree_levels.append(DendriticBranchNode(n_neurons * curr_branches))

    def forward(self, x):
        # x: (batch, in_features)
        batch = x.shape[0]
        
        # 1. Vectorized Leaf Stage
        # Select indices: (batch, n_neurons, n_leaves, leaf_size)
        leaf_in = x[:, self.indices] 
        # Apply leaf weights: (batch, n_neurons, n_leaves)
        leaves = torch.einsum('bnls,nls->bnl', leaf_in, self.leaf_w) + self.leaf_b
        
        # 2. Vectorized Dendritic Stages (Nested Branches)
        current = leaves
        level_acts = [current]
        
        for level_node in self.tree_levels:
            # Pair children: (batch, n_neurons, next_n, 2)
            b, n, l = current.shape
            current = current.view(b, n, l//2, 2)
            x_val, y_val = current[..., 0], current[..., 1]
            
            # Flatten neuron+branch for the vectorized node
            x_val = x_val.reshape(b, -1)
            y_val = y_val.reshape(b, -1)
            
            out = level_node(x_val, y_val)
            current = out.view(b, n, l//2)
            level_acts.append(current)
            
        return current.squeeze(-1), level_acts # (batch, n_neurons), [levels]


class DendriticLayer(nn.Module):
    """ Vectorized layer of Dendritic neurons with Residuals """
    def __init__(self, in_features: int, out_features: int, depth: int = 2):
        super().__init__()
        self.tree = DendriticTree(in_features, out_features, depth)
        self.soma_gain = nn.Parameter(torch.ones(1, out_features))
        self.soma_bias = nn.Parameter(torch.zeros(1, out_features))
        
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        res = self.skip(x)
        root, _ = self.tree(x)
        # soma: (batch, out_features)
        soma = F.leaky_relu(self.soma_gain * root + self.soma_bias, 0.1)
        return self.norm(soma + res)

class DendriticDEQCell(nn.Module):
    """
    Recurrent Dendritic Cell for Implicit Equilibrium.
    z_{k+1} = activation(LayerNorm(DendriticTree(z)) + x)
    This creates an infinite-depth dendritic structure.
    """
    def __init__(self, dim, depth=2):
        super().__init__()
        # The tree operates on the equilibrium state z
        self.tree = DendriticTree(dim, dim, depth)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, z, x):
        # Inject the constant input x directly into the dendritic tree's roots
        # This guarantees the equilibrium state is heavily dependent on the input image
        tree_out, _ = self.tree(z + x)
        
        # Apply normalization and non-linearity
        return torch.tanh(self.norm(tree_out))

class DendriticDEQLayer(nn.Module):
    """
    Infinite-depth Dendritic Layer using the 3-Phase Hybrid Solver.
    
    Phase 1: PJWR + Shanks → massively parallel GPU warm-up
    Phase 2: Block-Parallel Anderson → GPU-parallel AA across dim blocks
    Phase 3: Broyden → superlinear local convergence refinement
    """
    def __init__(self, in_dim, out_dim, depth=2, max_iter=50):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.cell = DendriticDEQCell(out_dim, depth)
        # 3-Phase Hybrid: PJWR → Block-Parallel Anderson → Broyden
        self.deq = DEQModule(self.cell,
                             max_iter=max_iter,
                             tol=1e-5,
                             # Phase 1: PJWR
                             pjwr_iters=8,
                             use_shanks=True,
                             # Phase 2: Block-Parallel Anderson
                             anderson_m=5,
                             anderson_beta=1.0,
                             n_blocks=4,
                             # Phase 3: Broyden
                             broyden_memory=15,
                             switch_tol=1e-2)
        
    def forward(self, x):
        x = self.proj(x)
        z_star = self.deq(x)
        return z_star
    
    def jacobian_penalty(self, x):
        """Compute Jacobian spectral norm penalty for this DEQ layer."""
        x_proj = self.proj(x)
        with torch.no_grad():
            z_star = self.deq(x_proj)
        return jacobian_spectral_norm(self.cell, z_star, x_proj, n_power_iters=5, target=0.9)


class DendriticNetwork(nn.Module):
    """Deep nested Dendritic network with optional DEQ mode."""
    def __init__(self, in_features, hidden_sizes, n_classes, depth=2, use_deq=False):
        super().__init__()
        self.use_deq = use_deq
        self.init_proj = nn.Linear(in_features, hidden_sizes[0])
        
        layers = []
        prev = hidden_sizes[0]
        for h in hidden_sizes:
            if use_deq:
                layers.append(DendriticDEQLayer(prev, h, depth))
            else:
                layers.append(DendriticLayer(prev, h, depth))
            prev = h
            
        self.body = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.init_proj(x)
        x = self.body(x)
        return self.classifier(x)
    
    def get_jacobian_penalty(self, x):
        """Sum Jacobian regularization penalty across all DEQ layers."""
        if not self.use_deq:
            return torch.tensor(0.0, device=x.device)
        penalty = torch.tensor(0.0, device=x.device)
        x_flat = x.flatten(1)
        h = self.init_proj(x_flat)
        for layer in self.body:
            if isinstance(layer, DendriticDEQLayer):
                penalty = penalty + layer.jacobian_penalty(h)
            h = layer(h)
        return penalty


class PointNeuronNetwork(nn.Module):
    """MLP baseline."""
    def __init__(self, in_features, hidden_sizes, n_classes):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            prev = h
        self.body       = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x):
        return self.classifier(self.body(x.flatten(1)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

def experiment_xor():
    print("\n" + "="*60)
    print("EXPERIMENT A: XOR — Point Neuron vs Hierarchical Dendritic")
    print("="*60)

    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]).repeat(200, 1)
    y = torch.tensor([0.,1.,1.,0.]).repeat(200)
    X = X + 0.1 * torch.randn_like(X)

    def run(model, name, is_dend, epochs=1500, lr=0.01):
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X)[0] if is_dend else model(X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = model(X)[0] if is_dend else model(X).squeeze(-1)
            acc = ((out > 0).float() == y).float().mean().item()
        print(f"  {name:<35} acc={acc*100:.1f}%")

    run(nn.Linear(2, 1),           "Point neuron (Linear only)", False)
    run(DendriticNeuron(2, depth=1),"Dendritic (depth=1)", True)
    run(DendriticNeuron(2, depth=2),"Dendritic (depth=2)", True)


def experiment_branches():
    print("\n" + "="*60)
    print("EXPERIMENT B: Branch Landscape (2-Moons)")
    print("="*60)

    try:
        from sklearn.datasets import make_moons
        X_np, y_np = make_moons(500, noise=0.15, random_state=42)
    except:
        t = np.linspace(0, math.pi, 250)
        X_np = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1-np.cos(t), 0.5-np.sin(t)]]) + 0.1*np.random.randn(500,2)
        y_np = np.array([0]*250 + [1]*250)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    model = DendriticNeuron(2, depth=2)
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(1000):
        opt.zero_grad()
        out, _ = model(X)
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()

    with torch.no_grad():
        out, _ = model(X)
        acc = ((out>0).float()==y).float().mean().item()
        print(f"  Moons accuracy: {acc*100:.1f}%")

    xx,yy = np.meshgrid(np.linspace(-2.5,3.5,100), np.linspace(-1.5,2.5,100))
    grid = torch.tensor(np.c_[xx.ravel(),yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        soma_out, lvl_acts = model(grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].contourf(xx, yy, (soma_out>0).numpy().reshape(100,100), alpha=.4, cmap="RdBu")
    axes[0].scatter(X_np[:,0], X_np[:,1], c=y_np, cmap="RdBu", s=5)
    axes[0].set_title("Decision Boundary")
    
    im1 = axes[1].contourf(xx, yy, soma_out.numpy().reshape(100,100), levels=20, cmap="viridis")
    plt.colorbar(im1, ax=axes[1]); axes[1].set_title("Soma Activation")
    
    avg_leaf = lvl_acts[0].mean(-1).numpy().reshape(100,100)
    im2 = axes[2].contourf(xx, yy, avg_leaf, levels=20, cmap="plasma")
    plt.colorbar(im2, ax=axes[2]); axes[2].set_title("Avg Leaf Activation")

    plt.tight_layout()
    path = "/home/moelanoby/.gemini/antigravity/brain/5365e7d6-5bcf-4aaa-a072-d81ccbb3ec2e/dendritic_branch_viz.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved → {path}")
    return path


def experiment_depth_ablation():
    print("\n" + "="*60)
    print("EXPERIMENT C: Depth Ablation (2-Spiral)")
    print("="*60)

    def make_spirals(n=500, noise=0.3):
        t = torch.linspace(0, 4*math.pi, n//2)
        r = t / (4*math.pi)
        X = torch.cat([torch.stack([r*torch.cos(t), r*torch.sin(t)], 1), 
                       torch.stack([r*torch.cos(t+math.pi), r*torch.sin(t+math.pi)], 1)]) + noise*torch.randn(n,2)
        y = torch.cat([torch.zeros(n//2), torch.ones(n//2)])
        return X, y

    X, y = make_spirals()
    for d in range(4):
        model = DendriticNeuron(2, depth=d)
        opt = optim.Adam(model.parameters(), lr=0.01)
        for _ in range(2000):
            opt.zero_grad()
            out, _ = model(X)
            F.binary_cross_entropy_with_logits(out, y).backward()
            opt.step()
        with torch.no_grad():
            out, _ = model(X)
            acc = ((out>0).float()==y).float().mean().item()
        print(f"  depth={d}  acc={acc*100:.1f}%")


def experiment_mnist():
    print("\n" + "="*60)
    print("EXPERIMENT D: MNIST Benchmark")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr_dl = DataLoader(datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf), 256, shuffle=True)
    te_dl = DataLoader(datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf), 512)

    # Matching the Dendritic parameter count
    # Dendritic: [64, 32] 
    # MLP: [64, 32]
    
    configs = [
        ("Hybrid-DEQ-Dendritic", DendriticNetwork(784, [128], 10, depth=2, use_deq=True)),
    ]

    # Jacobian regularization weight (lambda)
    lambda_jac = 0.01

    for name, model in configs:
        model = model.to(device)
        n_p = sum(p.numel() for p in model.parameters())
        print(f"\n  [{name}]  Params: {n_p:,}")
        print(f"  Solver: PJWR→Anderson→Broyden (3-Phase) | Jacobian Reg: λ={lambda_jac}")
        
        lr = 1e-3 if "DEQ" in name else 2e-3
        opt = optim.Adam(model.parameters(), lr=lr)
        
        for ep in range(10):
            model.train()
            epoch_jac_penalty = 0.0
            n_batches = 0
            for x, yb in tr_dl:
                x, yb = x.to(device), yb.to(device)
                opt.zero_grad()
                
                # Task loss
                logits = model(x)
                task_loss = F.cross_entropy(logits, yb)
                
                # Jacobian regularization (enforce contraction mapping)
                if model.use_deq and ep < 8:  # Warm-up: regularize early epochs
                    jac_penalty = model.get_jacobian_penalty(x)
                    loss = task_loss + lambda_jac * jac_penalty
                    epoch_jac_penalty += jac_penalty.item()
                else:
                    loss = task_loss
                
                loss.backward()
                # Gradient clipping for stability with implicit differentiation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                n_batches += 1
            
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, yb in te_dl:
                    x, yb = x.to(device), yb.to(device)
                    correct += (model(x).argmax(1)==yb).sum().item(); total += len(yb)
            
            jac_str = f" | Jac Pen: {epoch_jac_penalty/max(1,n_batches):.4f}" if model.use_deq and ep < 8 else ""
            print(f"    Epoch {ep+1:2d} | Test Acc: {correct/total*100:.2f}%{jac_str}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Hierarchical Dendritic Tree — True Branching Explorer      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    torch.manual_seed(42); np.random.seed(42)

    experiment_xor()
    branch_img = experiment_branches()
    experiment_depth_ablation()
    if HAS_TORCHVISION: experiment_mnist()
    print("\n✓ MNIST experiment complete with True Hierarchical logic.")
