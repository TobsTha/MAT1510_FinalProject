"""
Investigate why effective rank is not changing during training.

Hypotheses:
1. Lazy training regime - only last layer is being updated
2. Numerical precision issues in erank computation
3. Embeddings are actually changing but erank stays stable
4. The network initialized in a low-rank subspace and stays there
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from DD_R32_D48_training import (
    ReLUNetwork, create_task_matrix, generate_data,
    compute_gram_effective_rank, get_penultimate_embeddings, effective_rank, device
)

def analyze_layer_gradients(model, X, Y):
    """Check which layers have significant gradients."""
    criterion = nn.MSELoss()
    model.zero_grad()
    loss = criterion(model(X), Y)
    loss.backward()
    
    print("\nLayer-wise gradient analysis:")
    print("-" * 60)
    
    layer_idx = 0
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.norm().item()
            relative_grad = grad_norm / (weight_norm + 1e-10)
            grad_norms.append((layer_idx, name, grad_norm, relative_grad))
            layer_idx += 1
    
    # Sort by gradient magnitude
    grad_norms.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Layer':<40} {'Grad Norm':<12} {'Rel. Grad':<12}")
    print("-" * 60)
    for idx, name, grad_norm, rel_grad in grad_norms[:10]:
        print(f"{name:<40} {grad_norm:<12.6f} {rel_grad:<12.6f}")
    
    return grad_norms

def analyze_embeddings_directly(model, X):
    """Analyze the penultimate layer embeddings in detail."""
    embeddings = get_penultimate_embeddings(model, X)
    
    print("\nEmbedding Analysis:")
    print("-" * 60)
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean().item():.6f}")
    print(f"Std: {embeddings.std().item():.6f}")
    print(f"Min: {embeddings.min().item():.6f}")
    print(f"Max: {embeddings.max().item():.6f}")
    
    # How many neurons are "dead" (always zero due to ReLU)?
    dead_neurons = (embeddings.abs().max(dim=0).values < 1e-6).sum().item()
    print(f"Dead neurons (always ~0): {dead_neurons} / {embeddings.shape[1]}")
    
    # Singular value analysis
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    print(f"\nSingular value spectrum:")
    print(f"  Top 5: {S[:5].tolist()}")
    print(f"  Bottom 5: {S[-5:].tolist()}")
    print(f"  Ratio S[0]/S[-1]: {S[0].item() / (S[-1].item() + 1e-10):.2f}")
    
    # What fraction of variance is in top-k components?
    var_total = (S ** 2).sum()
    for k in [1, 2, 3, 5, 10, 20]:
        var_k = (S[:k] ** 2).sum()
        print(f"  Variance in top-{k}: {100 * var_k / var_total:.1f}%")
    
    return embeddings, S

def get_embedding_stats(model, X):
    """Get key statistics about embeddings."""
    embeddings = get_penultimate_embeddings(model, X)
    
    # Dead neurons
    dead_neurons = (embeddings.abs().max(dim=0).values < 1e-6).sum().item()
    total_neurons = embeddings.shape[1]
    
    # Gram erank
    erank = compute_gram_effective_rank(model, X)
    
    # Singular values
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    var_total = (S ** 2).sum()
    var_top1 = (S[:1] ** 2).sum() / var_total * 100
    var_top5 = (S[:5] ** 2).sum() / var_total * 100
    
    return {
        'erank': erank,
        'dead_neurons': dead_neurons,
        'total_neurons': total_neurons,
        'var_top1': var_top1.item(),
        'var_top5': var_top5.item(),
        'embeddings': embeddings,
        'singular_values': S
    }

def compare_init_vs_trained(checkpoint_path, checkpoint_100k_path=None, dim=64, depth=48, seed=42):
    """Compare random init vs trained model embeddings."""
    print("\n" + "=" * 60)
    print("COMPARING RANDOM INIT vs TRAINED MODEL")
    print("=" * 60)
    
    # Create task and data
    W = create_task_matrix(dim, rank=32, seed=seed)
    X, Y = generate_data(W, n_samples=128, seed=seed + 1)
    
    # Random init model
    torch.manual_seed(seed)
    model_init = ReLUNetwork(dim=dim, depth=depth)
    
    print("\n--- RANDOM INITIALIZATION ---")
    emb_init, S_init = analyze_embeddings_directly(model_init, X)
    erank_init = compute_gram_effective_rank(model_init, X)
    print(f"\nGram matrix effective rank: {erank_init:.4f}")
    stats_init = get_embedding_stats(model_init, X)
    
    stats_100k = None
    stats_200k = None
    
    # Check for 100k checkpoint
    if checkpoint_100k_path and os.path.exists(checkpoint_100k_path):
        print("\n--- MODEL AT 100K EPOCHS ---")
        checkpoint_100k = torch.load(checkpoint_100k_path, map_location=device)
        model_100k = ReLUNetwork(dim=dim, depth=depth)
        model_100k.load_state_dict(checkpoint_100k['model_state_dict'])
        emb_100k, S_100k = analyze_embeddings_directly(model_100k, X)
        erank_100k = compute_gram_effective_rank(model_100k, X)
        print(f"\nGram matrix effective rank: {erank_100k:.4f}")
        stats_100k = get_embedding_stats(model_100k, X)
    
    # Trained model (200k)
    if os.path.exists(checkpoint_path):
        print("\n--- TRAINED MODEL (200K EPOCHS) ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_trained = ReLUNetwork(dim=dim, depth=depth)
        model_trained.load_state_dict(checkpoint['model_state_dict'])
        
        emb_trained, S_trained = analyze_embeddings_directly(model_trained, X)
        erank_trained = compute_gram_effective_rank(model_trained, X)
        print(f"\nGram matrix effective rank: {erank_trained:.4f}")
        stats_200k = get_embedding_stats(model_trained, X)
        
        # Compare embeddings
        print("\n--- COMPARISON ---")
        emb_diff = (emb_trained - stats_init['embeddings']).norm() / stats_init['embeddings'].norm()
        print(f"Relative embedding change: {emb_diff:.4f}")
        
        # Correlation between embeddings
        emb_init_flat = stats_init['embeddings'].flatten()
        emb_trained_flat = emb_trained.flatten()
        corr = torch.corrcoef(torch.stack([emb_init_flat, emb_trained_flat]))[0, 1]
        print(f"Embedding correlation: {corr:.4f}")
        
        # Gradient analysis
        print("\n--- GRADIENT ANALYSIS (trained model) ---")
        analyze_layer_gradients(model_trained, X, Y)
        
        # Print comparison table
        print("\n" + "=" * 70)
        print("SUMMARY TABLE: Embedding Statistics Across Training")
        print("=" * 70)
        
        if stats_100k:
            print(f"\n{'Metric':<25} {'Random Init':<15} {'100k Epochs':<15} {'200k Epochs':<15}")
            print("-" * 70)
            print(f"{'Gram erank':<25} {stats_init['erank']:<15.2f} {stats_100k['erank']:<15.2f} {stats_200k['erank']:<15.2f}")
            dead_init = f"{stats_init['dead_neurons']}/{stats_init['total_neurons']}"
            dead_100k = f"{stats_100k['dead_neurons']}/{stats_100k['total_neurons']}"
            dead_200k = f"{stats_200k['dead_neurons']}/{stats_200k['total_neurons']}"
            print(f"{'Dead neurons':<25} {dead_init:<15} {dead_100k:<15} {dead_200k:<15}")
            var1_init = f"{stats_init['var_top1']:.1f}%"
            var1_100k = f"{stats_100k['var_top1']:.1f}%"
            var1_200k = f"{stats_200k['var_top1']:.1f}%"
            print(f"{'Variance in top-1 SV':<25} {var1_init:<15} {var1_100k:<15} {var1_200k:<15}")
            var5_init = f"{stats_init['var_top5']:.1f}%"
            var5_100k = f"{stats_100k['var_top5']:.1f}%"
            var5_200k = f"{stats_200k['var_top5']:.1f}%"
            print(f"{'Variance in top-5 SV':<25} {var5_init:<15} {var5_100k:<15} {var5_200k:<15}")
        else:
            print(f"\n{'Metric':<25} {'Random Init':<15} {'200k Epochs':<15}")
            print("-" * 55)
            print(f"{'Gram erank':<25} {stats_init['erank']:<15.2f} {stats_200k['erank']:<15.2f}")
            dead_init = f"{stats_init['dead_neurons']}/{stats_init['total_neurons']}"
            dead_200k = f"{stats_200k['dead_neurons']}/{stats_200k['total_neurons']}"
            print(f"{'Dead neurons':<25} {dead_init:<15} {dead_200k:<15}")
            var1_init = f"{stats_init['var_top1']:.1f}%"
            var1_200k = f"{stats_200k['var_top1']:.1f}%"
            print(f"{'Variance in top-1 SV':<25} {var1_init:<15} {var1_200k:<15}")
            var5_init = f"{stats_init['var_top5']:.1f}%"
            var5_200k = f"{stats_200k['var_top5']:.1f}%"
            print(f"{'Variance in top-5 SV':<25} {var5_init:<15} {var5_200k:<15}")
            print("\nNote: 100k checkpoint not available. To include it, save checkpoint at 100k epochs.")
        
        return model_init, model_trained, X, Y
    else:
        print(f"\nCheckpoint not found: {checkpoint_path}")
        return model_init, None, X, Y

def test_erank_sensitivity():
    """Test if erank is sensitive to small changes in embeddings."""
    print("\n" + "=" * 60)
    print("ERANK SENSITIVITY TEST")
    print("=" * 60)
    
    n, d = 128, 64
    
    # Create low-rank embeddings (similar to what we observe)
    torch.manual_seed(42)
    base_emb = torch.randn(n, 2) @ torch.randn(2, d)  # rank ~2
    
    base_erank = effective_rank(base_emb @ base_emb.T)
    print(f"\nBase embeddings (rank ~2): erank = {base_erank:.4f}")
    
    # Add noise of different magnitudes
    for noise_scale in [0.001, 0.01, 0.1, 0.5, 1.0]:
        noise = torch.randn_like(base_emb) * noise_scale * base_emb.std()
        perturbed = base_emb + noise
        erank_perturbed = effective_rank(perturbed @ perturbed.T)
        print(f"  + {noise_scale:.3f}*std noise: erank = {erank_perturbed:.4f}")

def check_weight_changes(checkpoint_path, dim=64, depth=48, seed=42):
    """Check how much weights have changed from initialization."""
    print("\n" + "=" * 60)
    print("WEIGHT CHANGE ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found")
        return
    
    # Random init
    torch.manual_seed(seed)
    model_init = ReLUNetwork(dim=dim, depth=depth)
    
    # Trained
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_trained = ReLUNetwork(dim=dim, depth=depth)
    model_trained.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n{'Layer':<40} {'Init Norm':<12} {'Trained Norm':<12} {'Change':<12} {'Rel Change':<12}")
    print("-" * 90)
    
    init_params = dict(model_init.named_parameters())
    trained_params = dict(model_trained.named_parameters())
    
    for name in init_params:
        init_norm = init_params[name].norm().item()
        trained_norm = trained_params[name].norm().item()
        diff_norm = (trained_params[name] - init_params[name]).norm().item()
        rel_change = diff_norm / (init_norm + 1e-10)
        
        print(f"{name:<40} {init_norm:<12.4f} {trained_norm:<12.4f} {diff_norm:<12.4f} {rel_change:<12.4f}")

if __name__ == "__main__":
    checkpoint_path = os.path.join(SCRIPT_DIR, 'DD_R32_D48_checkpoint.pt')
    
    # Run all analyses
    test_erank_sensitivity()
    check_weight_changes(checkpoint_path)
    model_init, model_trained, X, Y = compare_init_vs_trained(checkpoint_path)
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If the effective rank is ~2.4 and not changing:
1. The network likely initialized with most neurons dead (ReLU killing gradients)
2. Only a small subspace of the penultimate layer is active
3. Training is happening in this low-dimensional subspace
4. This is consistent with "lazy training" in very deep networks

Possible solutions to explore:
1. Different initialization (smaller weights to keep more neurons alive)
2. Different activation (LeakyReLU, GELU)
3. Batch normalization
4. Residual connections
5. Lower depth
""")
