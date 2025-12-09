"""
DD (Double Descent) Hypothesis Testing:
Can deep networks eventually fit high-rank functions with extended training?

Tests whether a deep network (depth=48) can eventually converge to low loss
on a high-rank task (rank=32) given enough training epochs.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import os
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to script directory
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
LOG_FILE = os.path.join(SCRIPT_DIR, 'DD_R32_D48_training.log')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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

sys.stdout = Logger(LOG_FILE)

# Device configuration - using CPU as it's faster for small models
device = torch.device('cpu')
print(f"Using device: {device}")


class ReLUNetwork(nn.Module):
    """Deep ReLU network with Kaiming initialization."""
    def __init__(self, dim=64, depth=2):
        super().__init__()
        layers = []
        for i in range(depth):
            layer = nn.Linear(dim, dim, bias=True)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < depth - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_task_matrix(dim, rank, seed):
    """Create a task matrix W with specified rank."""
    torch.manual_seed(seed)
    if rank >= dim:
        W = torch.randn(dim, dim)
    else:
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


def effective_rank(W):
    """
    Compute the effective rank of a matrix W.
    Effective rank = exp(entropy of normalized singular values)
    """
    if W.dim() > 2:
        W = W.view(W.size(0), -1)
    
    s = torch.linalg.svdvals(W.float())
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    
    s_hat = s / s.sum()
    entropy = -(s_hat * torch.log(s_hat + 1e-10)).sum()
    return torch.exp(entropy).item()


def get_penultimate_embeddings(model, X):
    """Get embeddings from penultimate layer (before final linear)."""
    model.eval()
    layers = list(model.network.children())
    with torch.no_grad():
        x = X
        for layer in layers[:-1]:
            x = layer(x)
    return x


def compute_gram_effective_rank(model, X):
    """Compute effective rank of Gram matrix K = Phi(X) @ Phi(X).T"""
    embeddings = get_penultimate_embeddings(model, X)
    gram = embeddings @ embeddings.T
    return effective_rank(gram)


def compute_variance_spectrum(model, X):
    """
    Compute variance explained by each singular value of penultimate embeddings.
    Returns singular values and cumulative variance explained.
    """
    embeddings = get_penultimate_embeddings(model, X)
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    
    # Compute variance (squared singular values)
    variance = S ** 2
    total_variance = variance.sum()
    
    # Cumulative variance explained (as percentages)
    cumulative_variance = torch.cumsum(variance, dim=0) / total_variance * 100
    
    return {
        'singular_values': S.tolist(),
        'variance': variance.tolist(),
        'cumulative_variance_pct': cumulative_variance.tolist(),
        'total_variance': total_variance.item()
    }


def train_with_tracking(model, X, Y, epochs, lr, log_every=100, print_every=1000,
                        checkpoint_every=10000, checkpoint_path=None,
                        use_scheduler=True, start_epoch=0, erank_every=1000,
                        variance_every=10000, model_checkpoint_every=20000):
    """
    Train model and track loss, effective rank, and variance spectrum over time.
    
    Features:
    - Loss tracking for plotting
    - Effective rank tracking every erank_every epochs
    - Variance spectrum tracking every variance_every epochs (saved to data/)
    - Model checkpointing for resumable training
    - Separate model snapshots every model_checkpoint_every epochs (saved to checkpoints/)
    - Optional LR scheduler (ReduceLROnPlateau)
    
    Args:
        variance_every: Save variance spectrum every N epochs (default: 10000)
        model_checkpoint_every: Save separate model snapshot every N epochs (default: 20000)
    
    Returns:
        loss_history: list of (epoch, loss) tuples
        erank_history: list of (epoch, effective_rank) tuples
        variance_history: list of (epoch, variance_data) tuples
        final_loss: final training loss
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'DD_R32_D48_checkpoint.pt')
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()
    
    # LR scheduler - reduce LR when loss plateaus
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5000, min_lr=1e-6
        )
    
    loss_history = []
    erank_history = []
    variance_history = []
    start_time = time.time()
    
    # Compute initial effective rank and variance spectrum
    erank = compute_gram_effective_rank(model, X)
    erank_history.append((start_epoch, erank))
    print(f"  Initial effective rank: {erank:.2f}")
    
    # Compute initial variance spectrum
    model.eval()
    variance_data = compute_variance_spectrum(model, X)
    variance_history.append((start_epoch, variance_data))
    save_variance_spectrum(start_epoch, variance_data)
    print(f"  Initial variance spectrum saved")
    
    model.train()
    for epoch in range(start_epoch, start_epoch + epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(loss_val)
        
        # Check for divergence
        if not np.isfinite(loss_val):
            print(f"  Training diverged at epoch {epoch}")
            break
        
        # Log loss
        if epoch % log_every == 0:
            loss_history.append((epoch, loss_val))
        
        # Compute and log effective rank
        if epoch % erank_every == 0 and epoch > start_epoch:
            model.eval()
            erank = compute_gram_effective_rank(model, X)
            erank_history.append((epoch, erank))
            model.train()
        
        # Compute and save variance spectrum every variance_every epochs
        if epoch % variance_every == 0 and epoch > start_epoch:
            model.eval()
            variance_data = compute_variance_spectrum(model, X)
            variance_history.append((epoch, variance_data))
            save_variance_spectrum(epoch, variance_data)
            print(f"  [Variance spectrum saved at epoch {epoch}]")
            model.train()
        
        # Save separate model checkpoint every model_checkpoint_every epochs
        if epoch % model_checkpoint_every == 0 and epoch > start_epoch:
            model_snapshot_path = os.path.join(CHECKPOINTS_DIR, f'DD_R32_D48_model_epoch_{epoch}.pt')
            save_model_snapshot(model, epoch, loss_val, erank_history[-1][1] if erank_history else 0, model_snapshot_path)
            print(f"  [Model snapshot saved at epoch {epoch}]")
        
        # Print progress
        if epoch % print_every == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch - start_epoch + 1)) * (epochs - (epoch - start_epoch) - 1)
            current_lr = optimizer.param_groups[0]['lr']
            # Get latest erank
            latest_erank = erank_history[-1][1] if erank_history else 0
            print(f"  Epoch {epoch:6d}: loss = {loss_val:.6f}, erank = {latest_erank:.2f}, lr = {current_lr:.6f} "
                  f"(elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")
        
        # Save resumable checkpoint
        if epoch % checkpoint_every == 0 and epoch > start_epoch:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history, checkpoint_path, erank_history, variance_history)
            print(f"  [Checkpoint saved at epoch {epoch}]")
    
    # Final effective rank
    model.eval()
    erank = compute_gram_effective_rank(model, X)
    erank_history.append((start_epoch + epochs, erank))
    
    # Final variance spectrum
    variance_data = compute_variance_spectrum(model, X)
    variance_history.append((start_epoch + epochs, variance_data))
    save_variance_spectrum(start_epoch + epochs, variance_data)
    
    # Final model snapshot
    final_epoch = start_epoch + epochs
    model_snapshot_path = os.path.join(CHECKPOINTS_DIR, f'DD_R32_D48_model_epoch_{final_epoch}.pt')
    with torch.no_grad():
        output = model(X)
        final_loss = criterion(output, Y).item()
    save_model_snapshot(model, final_epoch, final_loss, erank, model_snapshot_path)
    
    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, final_epoch, loss_history, checkpoint_path, erank_history, variance_history)
    loss_history.append((final_epoch, final_loss))
    
    return loss_history, erank_history, variance_history, final_loss


def save_variance_spectrum(epoch, variance_data):
    """Save variance spectrum data to JSON file."""
    filepath = os.path.join(DATA_DIR, f'variance_spectrum_epoch_{epoch}.json')
    data = {
        'epoch': epoch,
        **variance_data
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_model_snapshot(model, epoch, loss, erank, path):
    """Save a standalone model snapshot for later analysis."""
    snapshot = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'erank': erank
    }
    torch.save(snapshot, path)


def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, path, erank_history=None, variance_history=None):
    """Save model checkpoint for resumable training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_history': loss_history,
        'erank_history': erank_history,
        'variance_history': variance_history
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint to resume training."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss_history', [])


def run_experiment(rank, depth, dim=64, n_samples=128, epochs=100000, 
                   learning_rates=[0.0001, 0.0002, 0.00005], seed=42):
    """
    Run long training experiment for a specific rank/depth configuration.
    """
    print("="*70)
    print(f"DD Hypothesis Testing: Rank={rank}, Depth={depth}")
    print("="*70)
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rates to try: {learning_rates}")
    print(f"  Dimension: {dim}")
    print(f"  Samples: {n_samples}")
    print("="*70)
    
    # Create task
    W = create_task_matrix(dim, rank, seed=seed)
    X, Y = generate_data(W, n_samples, seed=seed + 1)
    
    # Compute baseline loss (if network outputs zeros)
    baseline_loss = (Y ** 2).mean().item()
    print(f"\nBaseline loss (zero output): {baseline_loss:.2f}")
    
    results = {
        'rank': rank,
        'depth': depth,
        'dim': dim,
        'n_samples': n_samples,
        'epochs': epochs,
        'learning_rates': learning_rates,
        'seed': seed,
        'baseline_loss': baseline_loss,
        'experiments': []
    }
    
    best_overall_loss = float('inf')
    best_lr = None
    best_history = None
    
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Training with lr={lr}")
        print(f"{'='*70}")
        
        torch.manual_seed(seed + 100)
        model = ReLUNetwork(dim=dim, depth=depth)
        
        history, final_loss = train_with_tracking(
            model, X, Y, epochs=epochs, lr=lr,
            log_every=100, print_every=5000
        )
        
        exp_result = {
            'lr': lr,
            'final_loss': final_loss,
            'loss_history': history
        }
        results['experiments'].append(exp_result)
        
        print(f"\nFinal loss with lr={lr}: {final_loss:.6f}")
        
        if final_loss < best_overall_loss:
            best_overall_loss = final_loss
            best_lr = lr
            best_history = history
    
    results['best_lr'] = best_lr
    results['best_final_loss'] = best_overall_loss
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Best learning rate: {best_lr}")
    print(f"Best final loss: {best_overall_loss:.6f}")
    print(f"Baseline loss: {baseline_loss:.2f}")
    print(f"Loss reduction: {baseline_loss / best_overall_loss:.1f}x" if best_overall_loss > 0 else "N/A")
    
    return results


def save_results(results, filename):
    """Save results to JSON."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Configuration - using best LR found from testing (0.0002)
    RANK = 32
    DEPTH = 48
    EPOCHS = 200000
    LR = 0.0002  # Best LR from quick test
    
    print("="*70)
    print("DD Hypothesis Testing")
    print("Can deep networks eventually fit high-rank functions?")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Rank: {RANK}")
    print(f"  Depth: {DEPTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR} (with ReduceLROnPlateau scheduler)")
    print(f"  Resumable checkpoint: checkpoints/DD_R32_D48_checkpoint.pt")
    print(f"  Model snapshots: every 20,000 epochs in checkpoints/")
    print(f"  Variance spectra: every 10,000 epochs in data/")
    print("="*70)
    
    start_time = time.time()
    
    # Create task
    dim, n_samples, seed = 64, 128, 42
    W = create_task_matrix(dim, RANK, seed=seed)
    X, Y = generate_data(W, n_samples, seed=seed + 1)
    
    baseline_loss = (Y ** 2).mean().item()
    print(f"\nBaseline loss (zero output): {baseline_loss:.2f}")
    
    # Create model
    torch.manual_seed(seed + 100)
    model = ReLUNetwork(dim=dim, depth=DEPTH)
    
    # Train
    print("\nStarting training...")
    loss_history, erank_history, variance_history, final_loss = train_with_tracking(
        model, X, Y, 
        epochs=EPOCHS, 
        lr=LR,
        log_every=100,
        print_every=5000,
        checkpoint_every=10000,
        checkpoint_path=os.path.join(CHECKPOINTS_DIR, 'DD_R32_D48_checkpoint.pt'),
        use_scheduler=True,
        erank_every=5000,
        variance_every=10000,
        model_checkpoint_every=20000
    )
    
    total_time = time.time() - start_time
    
    # Save results
    final_erank = erank_history[-1][1] if erank_history else 0
    initial_erank = erank_history[0][1] if erank_history else 0
    
    results = {
        'rank': RANK,
        'depth': DEPTH,
        'dim': dim,
        'n_samples': n_samples,
        'epochs': EPOCHS,
        'lr': LR,
        'seed': seed,
        'baseline_loss': baseline_loss,
        'best_final_loss': final_loss,
        'best_lr': LR,
        'total_time_seconds': total_time,
        'initial_erank': initial_erank,
        'final_erank': final_erank,
        'erank_history': erank_history,
        'experiments': [{
            'lr': LR,
            'final_loss': final_loss,
            'loss_history': loss_history
        }]
    }
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Baseline loss: {baseline_loss:.2f}")
    print(f"Improvement: {baseline_loss / final_loss:.1f}x" if final_loss > 0 else "N/A")
    print(f"Initial effective rank: {initial_erank:.2f}")
    print(f"Final effective rank: {final_erank:.2f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, os.path.join(DATA_DIR, f"DD_R32_D48_results_{timestamp}.json"))
    save_results(results, os.path.join(DATA_DIR, "DD_R32_D48_results.json"))
    
    print(f"\nModel checkpoint saved to: {CHECKPOINTS_DIR}/DD_R32_D48_checkpoint.pt")
    print(f"Variance spectra saved to: {DATA_DIR}/variance_spectrum_epoch_*.json")
    print(f"Model snapshots saved to: {CHECKPOINTS_DIR}/DD_R32_D48_model_epoch_*.pt")
    print("\nRun 'notebooks/DD_R32_D48_plot.ipynb' to visualize the loss curves.")
