"""
Reproduction of Figure 1 from:
"The Low-Rank Simplicity Bias in Deep Networks" by Huh et al.

Figure 1: Training loss of neural networks of different depths optimized
to solve linear regression Y = XW, where W has varying rank [1, 4, 16, 32, 64].

Adapted parameters for ~15 min runtime on GTX 1660 Ti:
- 6000 epochs (instead of 24000)
- 2 task seeds x 2 init seeds = 4 runs (instead of 25)
- LR scaling heuristic: lr = base_lr / sqrt(depth), with 3 multipliers [0.5, 1, 2]
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from datetime import datetime

# Setup logging to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger('figure1_training.log')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class ReLUNetwork(nn.Module):
    """Deep ReLU network with Kaiming initialization."""
    def __init__(self, dim=64, depth=2):
        super().__init__()
        layers = []
        for i in range(depth):
            # Paper uses bias=False, but we add small bias init to help gradient flow
            layer = nn.Linear(dim, dim, bias=True)
            # Kaiming init with gain=sqrt(2) as per paper
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < depth - 1:  # No ReLU after last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LinearNetwork(nn.Module):
    """Deep LINEAR network (no activations) - for comparison."""
    def __init__(self, dim=64, depth=2):
        super().__init__()
        layers = []
        for i in range(depth):
            layer = nn.Linear(dim, dim, bias=False)
            # Small init for deep linear nets to avoid exploding
            nn.init.normal_(layer.weight, std=1.0/np.sqrt(dim * depth))
            layers.append(layer)
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_task_matrix(dim, rank, seed):
    """Create a task matrix W with specified rank."""
    torch.manual_seed(seed)
    if rank >= dim:
        W = torch.randn(dim, dim)
    else:
        # Low-rank matrix: W = U @ V^T where U, V are (dim, rank)
        U = torch.randn(dim, rank)
        V = torch.randn(dim, rank)
        W = U @ V.T / np.sqrt(rank)
    return W.to(device)


def generate_data(W, n_samples=128, seed=None):
    """Generate data X and targets Y = X @ W."""
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.randn(n_samples, W.shape[0]).to(device)
    Y = X @ W
    return X, Y


def train_model(model, X, Y, epochs, lr, batch_size=128, lr_step_epoch=None,
                early_stop_patience=150, early_stop_threshold=1e-7, diverge_threshold=1e6):
    """
    Train model with SGD + momentum, optional LR step, and early stopping.
    
    Early stopping triggers if:
    - Loss drops below early_stop_threshold (converged), OR
    - Loss doesn't improve for early_stop_patience epochs (plateaued), OR
    - Loss exceeds diverge_threshold (diverged)
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()
    
    n_samples = X.shape[0]
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        # LR step at specified epoch
        if lr_step_epoch is not None and epoch == lr_step_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 10
            # Reset patience after LR step
            patience_counter = 0
        
        # Full batch training (no shuffling needed for full batch)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        epoch_loss = loss.item()
        
        # Early stopping: divergence check
        if not np.isfinite(epoch_loss) or epoch_loss > diverge_threshold:
            return float('inf')  # Signal divergence
        
        # Early stopping: convergence check
        if epoch_loss < early_stop_threshold:
            break
        
        # Early stopping: patience check
        if epoch_loss < best_loss - 1e-9:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break
    
    actual_epochs = epoch + 1  # Store how many epochs were actually run
    
    # Compute final training loss
    model.eval()
    with torch.no_grad():
        output = model(X)
        final_loss = criterion(output, Y).item()
    
    return final_loss, actual_epochs


def get_learning_rates_for_depth(depth, base_lr=0.1, multipliers_shallow=[0.1, 0.25, 0.5, 1.0, 2.0]):
    """
    Compute learning rates for a given depth using the heuristic: lr ∝ 1/sqrt(d).
    For deeper networks (depth >= 16), use a wider range of multipliers.
    """
    scaled_lr = base_lr / np.sqrt(depth)
    
    if depth >= 32:
        # Very deep networks - try extremely small LRs too
        multipliers = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    elif depth >= 16:
        # Moderately deep networks
        multipliers = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    else:
        multipliers = multipliers_shallow
    
    return [scaled_lr * m for m in multipliers]


def run_experiment(depths, ranks, base_lr=0.15, lr_multipliers=[0.5, 1.0, 2.0],
                   n_task_seeds=2, n_init_seeds=2, epochs=6000, dim=64, n_samples=128):
    """Run the full experiment."""
    
    lr_step_epoch = int(epochs * 0.75)  # LR step at 75% of training
    
    results = {
        'depths': depths,
        'ranks': ranks,
        'base_lr': base_lr,
        'lr_multipliers': lr_multipliers,
        'n_task_seeds': n_task_seeds,
        'n_init_seeds': n_init_seeds,
        'epochs': epochs,
        'dim': dim,
        'n_samples': n_samples,
        'losses': {},  # losses[rank][depth] = list of best losses
        'best_lrs': {},  # best_lrs[rank][depth] = list of best LRs chosen
        'best_epochs': {}  # best_epochs[rank][depth] = list of epochs trained
    }
    
    total_configs = len(depths) * len(ranks)
    config_count = 0
    start_time = time.time()
    
    for rank in ranks:
        results['losses'][rank] = {}
        results['best_lrs'][rank] = {}
        results['best_epochs'][rank] = {}
        print(f"\n{'='*60}")
        print(f"Task Rank: {rank}")
        print(f"{'='*60}")
        
        for depth in depths:
            config_count += 1
            depth_start = time.time()
            best_losses = []
            best_lrs_for_config = []
            best_epochs_for_config = []
            
            # Get depth-scaled learning rates
            learning_rates = get_learning_rates_for_depth(depth, base_lr, lr_multipliers)
            
            for task_seed in range(n_task_seeds):
                # Create task matrix with this seed
                W = create_task_matrix(dim, rank, seed=task_seed * 1000)
                X, Y = generate_data(W, n_samples, seed=task_seed * 1000 + 1)
                
                for init_seed in range(n_init_seeds):
                    # Find best learning rate for this configuration
                    best_loss = float('inf')
                    best_lr = None
                    best_epochs = 0
                    
                    for lr in learning_rates:
                        torch.manual_seed(init_seed * 100 + 42)
                        model = ReLUNetwork(dim=dim, depth=depth)
                        
                        # Deep networks need more patience
                        patience = 300 if depth >= 16 else 150
                        
                        loss, n_epochs = train_model(model, X, Y, epochs=epochs, lr=lr, 
                                         lr_step_epoch=lr_step_epoch,
                                         early_stop_patience=patience)
                        
                        # Skip if training diverged (nan or inf)
                        if not np.isfinite(loss):
                            continue
                        
                        if loss < best_loss:
                            best_loss = loss
                            best_lr = lr
                            best_epochs = n_epochs
                    
                    # If all LRs diverged, use the smallest LR as fallback
                    if best_lr is None:
                        best_lr = min(learning_rates)
                        best_epochs = 0
                    
                    best_losses.append(best_loss)
                    best_lrs_for_config.append(best_lr)
                    best_epochs_for_config.append(best_epochs)
            
            results['losses'][rank][depth] = best_losses
            results['best_lrs'][rank][depth] = best_lrs_for_config
            results['best_epochs'][rank][depth] = best_epochs_for_config
            
            depth_time = time.time() - depth_start
            elapsed = time.time() - start_time
            remaining = (elapsed / config_count) * (total_configs - config_count)
            
            mean_loss = np.mean(best_losses)
            std_loss = np.std(best_losses)
            mean_lr = np.mean(best_lrs_for_config)
            mean_epochs = np.mean(best_epochs_for_config)
            
            print(f"  Depth {depth:2d}: loss = {mean_loss:.6f} +/- {std_loss:.6f}, "
                  f"best_lr ~ {mean_lr:.4f}, epochs = {mean_epochs:.0f} ({depth_time:.1f}s, ETA: {remaining/60:.1f}min)")
    
    total_time = time.time() - start_time
    results['total_time_seconds'] = total_time
    print(f"\n{'='*60}")
    print(f"Experiment completed in {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    return results


def print_best_lrs_summary(results):
    """Print a summary of the best learning rates found."""
    print("\n" + "="*70)
    print("BEST LEARNING RATES SUMMARY")
    print("="*70)
    print(f"LR heuristic: lr = {results['base_lr']} / sqrt(depth) * multiplier")
    print(f"Multipliers tried: {results['lr_multipliers']}")
    print("-"*70)
    
    depths = results['depths']
    ranks = results['ranks']
    
    # Header
    print(f"{'Depth':<8}", end='')
    print(f"{'Theory LR':<12}", end='')
    for rank in ranks:
        print(f"Rank={rank:<6}", end='')
    print()
    print("-"*70)
    
    for depth in depths:
        theory_lr = results['base_lr'] / np.sqrt(depth)
        print(f"{depth:<8}", end='')
        print(f"{theory_lr:<12.4f}", end='')
        for rank in ranks:
            # Handle both int and str keys (in case of JSON serialization)
            rank_key = rank if rank in results['best_lrs'] else str(rank)
            depth_key = depth if depth in results['best_lrs'][rank_key] else str(depth)
            mean_lr = np.mean(results['best_lrs'][rank_key][depth_key])
            print(f"{mean_lr:<10.4f}", end='')
        print()
    
    print("-"*70)
    print("Note: Values shown are mean best LR across seeds for each (depth, rank) config")
    print("="*70)


def save_results(results, filename):
    """Save results to JSON file."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("="*60)
    print("Reproducing Figure 1: Training Loss vs Depth for Different Task Ranks")
    print("from 'The Low-Rank Simplicity Bias in Deep Networks'")
    print("="*60)
    
    # Experiment parameters (adapted for ~15 min runtime)
    depths = [2, 4, 8, 16, 32, 48, 64]
    ranks = [1, 4, 16, 32, 64]
    
    # LR heuristic: lr = base_lr / sqrt(depth), try multipliers around it
    # Extended range to lower values for deeper networks that need smaller LRs
    base_lr = 0.1
    lr_multipliers = [0.1, 0.25, 0.5, 1.0, 2.0]
    
    n_task_seeds = 1
    n_init_seeds = 1
    epochs = 10000  # Early stopping will halt converged models earlier
    dim = 64
    n_samples = 128
    
    print(f"\nConfiguration:")
    print(f"  Depths: {depths}")
    print(f"  Ranks: {ranks}")
    print(f"  LR heuristic: {base_lr} / sqrt(depth) * {lr_multipliers}")
    print(f"  Seeds: {n_task_seeds} task x {n_init_seeds} init = {n_task_seeds * n_init_seeds} runs per config")
    print(f"  Epochs: {epochs}")
    print(f"  Dimension: {dim}")
    print(f"  Samples: {n_samples}")
    print(f"\nTotal training runs: {len(depths) * len(ranks) * n_task_seeds * n_init_seeds * len(lr_multipliers)}")
    
    # Run experiment
    results = run_experiment(
        depths=depths,
        ranks=ranks,
        base_lr=base_lr,
        lr_multipliers=lr_multipliers,
        n_task_seeds=n_task_seeds,
        n_init_seeds=n_init_seeds,
        epochs=epochs,
        dim=dim,
        n_samples=n_samples
    )
    
    # Print best LRs summary
    print_best_lrs_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"figure1_results_{timestamp}.json")
    save_results(results, "figure1_results.json")  # Also save without timestamp for easy access
    
    print("\nRun the Jupyter notebook 'figure1_plot.ipynb' to generate the figure.")
