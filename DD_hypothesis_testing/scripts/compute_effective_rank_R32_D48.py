"""
Compute the effective rank of the trained network's embeddings.

Effective rank is computed over the Gram matrix of the penultimate layer embeddings,
as described in "The Low-Rank Simplicity Bias in Deep Networks".

Effective rank = exp(entropy of normalized singular values)
"""

import torch
import torch.nn as nn
import numpy as np

# Import from main script
from DD_R32_D48_training import ReLUNetwork, create_task_matrix, generate_data, device


def get_penultimate_embeddings(model, X):
    """
    Get embeddings from the penultimate layer (before final linear layer).
    """
    model.eval()
    
    # Get all layers
    layers = list(model.network.children())
    
    # Penultimate layer is everything except the last Linear layer
    # Network structure: Linear, ReLU, Linear, ReLU, ..., Linear (last)
    # We want output after the second-to-last ReLU
    
    with torch.no_grad():
        x = X
        # Go through all layers except the last one
        for layer in layers[:-1]:
            x = layer(x)
    
    return x


def effective_rank(W):
    """
    Compute the effective rank of a matrix W.
    Effective rank = exp(entropy of normalized singular values)
    
    From the paper:
    erank(W) = exp(H(s_hat)) where s_hat = s / sum(s) and H is entropy
    """
    # Compute SVD
    if W.dim() > 2:
        W = W.view(W.size(0), -1)
    
    s = torch.linalg.svdvals(W.float())
    
    # Remove near-zero singular values for numerical stability
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    
    # Normalize singular values
    s_hat = s / s.sum()
    
    # Compute entropy
    entropy = -(s_hat * torch.log(s_hat + 1e-10)).sum()
    
    # Effective rank = exp(entropy)
    erank = torch.exp(entropy).item()
    
    return erank


def compute_gram_effective_rank(embeddings):
    """
    Compute effective rank of the Gram matrix K = Phi(X) @ Phi(X).T
    """
    # Gram matrix
    gram = embeddings @ embeddings.T
    return effective_rank(gram)


if __name__ == "__main__":
    print("="*60)
    print("Computing Effective Rank of Trained Network")
    print("="*60)
    
    # Configuration (must match training)
    RANK = 32
    DEPTH = 48
    dim, n_samples, seed = 64, 128, 42
    
    # Load checkpoint
    checkpoint_path = 'DD_R32_D48_checkpoint.pt'
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trained_epochs = checkpoint['epoch']
    print(f"Model trained for {trained_epochs} epochs")
    
    # Recreate model and load weights
    model = ReLUNetwork(dim=dim, depth=DEPTH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Recreate data (same seed as training)
    W = create_task_matrix(dim, RANK, seed=seed)
    X, Y = generate_data(W, n_samples, seed=seed + 1)
    
    # Get embeddings from penultimate layer
    print("\nComputing penultimate layer embeddings...")
    embeddings = get_penultimate_embeddings(model, X)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute effective rank of embeddings
    print("\nComputing effective rank of embeddings...")
    erank_embeddings = effective_rank(embeddings)
    print(f"Effective rank of embeddings Phi(X): {erank_embeddings:.2f}")
    
    # Compute effective rank of Gram matrix
    print("\nComputing effective rank of Gram matrix K = Phi(X) @ Phi(X).T...")
    erank_gram = compute_gram_effective_rank(embeddings)
    print(f"Effective rank of Gram matrix: {erank_gram:.2f}")
    
    # For comparison, compute effective rank of the task matrix W
    print("\nFor comparison:")
    erank_W = effective_rank(W)
    print(f"Effective rank of task matrix W (rank={RANK}): {erank_W:.2f}")
    
    # Compute effective rank of random initialization (for comparison)
    print("\nComputing effective rank at random initialization...")
    torch.manual_seed(seed + 100)
    model_init = ReLUNetwork(dim=dim, depth=DEPTH).to(device)
    embeddings_init = get_penultimate_embeddings(model_init, X)
    erank_init = compute_gram_effective_rank(embeddings_init)
    print(f"Effective rank of Gram matrix (random init): {erank_init:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Task: rank-{RANK} matrix approximation")
    print(f"Network: {DEPTH}-layer ReLU network")
    print(f"Training: {trained_epochs} epochs")
    print(f"\nEffective ranks (Gram matrix):")
    print(f"  At initialization: {erank_init:.2f}")
    print(f"  After training:    {erank_gram:.2f}")
    print(f"  Task matrix W:     {erank_W:.2f}")
    print(f"\nObservation: ", end="")
    if erank_gram < erank_init * 0.5:
        print("Network learned LOW-RANK representation (rank collapsed)")
    elif erank_gram > erank_init * 0.9:
        print("Network maintained HIGH-RANK representation")
    else:
        print("Network has INTERMEDIATE-RANK representation")
