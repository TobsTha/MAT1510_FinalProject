"""
Quick benchmark: CPU vs GPU for small model training.
Tests which is faster for the Figure 1 experiment setup.
"""

import torch
import torch.nn as nn
import time

def create_model(dim, depth, device):
    layers = []
    for i in range(depth):
        layer = nn.Linear(dim, dim, bias=True)
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < depth - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers).to(device)
    return model

def benchmark(device, dim=64, depth=8, n_samples=128, epochs=500):
    """Run training benchmark on specified device."""
    device = torch.device(device)
    
    # Create data
    X = torch.randn(n_samples, dim).to(device)
    Y = torch.randn(n_samples, dim).to(device)
    
    # Create model
    model = create_model(dim, depth, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

if __name__ == "__main__":
    print("="*60)
    print("CPU vs GPU Benchmark for Small Model Training")
    print("="*60)
    
    # Check GPU availability
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, skipping GPU benchmark")
    
    configs = [
        {"dim": 64, "depth": 4, "n_samples": 128, "epochs": 1000},
        {"dim": 64, "depth": 16, "n_samples": 128, "epochs": 1000},
        {"dim": 64, "depth": 64, "n_samples": 128, "epochs": 1000},
        {"dim": 256, "depth": 8, "n_samples": 512, "epochs": 500},  # Larger model for comparison
    ]
    
    print(f"\n{'Config':<40} {'CPU (s)':<12} {'GPU (s)':<12} {'Winner':<10}")
    print("-"*74)
    
    for cfg in configs:
        config_str = f"dim={cfg['dim']}, depth={cfg['depth']}, n={cfg['n_samples']}"
        
        # CPU benchmark
        cpu_time = benchmark('cpu', **cfg)
        
        # GPU benchmark
        if has_gpu:
            gpu_time = benchmark('cuda', **cfg)
            winner = "CPU" if cpu_time < gpu_time else "GPU"
            speedup = max(cpu_time, gpu_time) / min(cpu_time, gpu_time)
            winner_str = f"{winner} ({speedup:.1f}x)"
        else:
            gpu_time = float('nan')
            winner_str = "N/A"
        
        print(f"{config_str:<40} {cpu_time:<12.3f} {gpu_time:<12.3f} {winner_str:<10}")
    
    print("-"*74)
    print("\nConclusion: For small models (dim=64), CPU is often faster due to")
    print("GPU kernel launch overhead. GPU wins for larger models.")
