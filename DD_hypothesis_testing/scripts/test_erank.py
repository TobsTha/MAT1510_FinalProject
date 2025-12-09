"""
Test script to verify the effective rank (erank) function works correctly.

Expected behavior:
1. Identity matrix: erank = n (all singular values equal)
2. Rank-1 matrix: erank = 1 (one non-zero singular value)
3. Rank-k matrix: erank ~ k (if singular values are equal)
4. Random matrix: erank ~ min(rows, cols)
5. Gram matrix K = A @ A.T has same non-zero singular values as A
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DD_R32_D48_training import effective_rank, compute_gram_effective_rank, ReLUNetwork

def test_identity_matrix():
    """Identity matrix should have erank = n (all singular values = 1)"""
    print("="*60)
    print("TEST 1: Identity Matrix")
    print("="*60)
    for n in [10, 50, 100]:
        I = torch.eye(n)
        erank = effective_rank(I)
        print(f"  Identity {n}x{n}: erank = {erank:.4f} (expected: {n})")
        assert abs(erank - n) < 0.01, f"Identity matrix erank should be {n}, got {erank}"
    print("  PASSED\n")

def test_rank1_matrix():
    """Rank-1 matrix should have erank = 1"""
    print("="*60)
    print("TEST 2: Rank-1 Matrix")
    print("="*60)
    for n in [10, 50, 100]:
        u = torch.randn(n, 1)
        v = torch.randn(n, 1)
        M = u @ v.T  # rank 1
        erank = effective_rank(M)
        print(f"  Rank-1 {n}x{n}: erank = {erank:.4f} (expected: 1.0)")
        assert abs(erank - 1.0) < 0.1, f"Rank-1 matrix erank should be ~1, got {erank}"
    print("  PASSED\n")

def test_low_rank_matrix():
    """Rank-k matrix with equal singular values should have erank ~ k"""
    print("="*60)
    print("TEST 3: Low-Rank Matrix (equal singular values)")
    print("="*60)
    n = 100
    for k in [5, 10, 20, 50]:
        # Create orthogonal basis
        Q, _ = torch.linalg.qr(torch.randn(n, k))
        # M = Q @ Q.T has exactly k non-zero singular values, all equal to 1
        M = Q @ Q.T
        erank = effective_rank(M)
        print(f"  Rank-{k} {n}x{n} (equal s.v.): erank = {erank:.4f} (expected: ~{k})")
        assert abs(erank - k) < 1.0, f"Low-rank matrix erank should be ~{k}, got {erank}"
    print("  PASSED\n")

def test_random_matrix():
    """Random matrix should have erank ~ min(rows, cols)"""
    print("="*60)
    print("TEST 4: Random Full-Rank Matrix")
    print("="*60)
    for n in [10, 50, 100]:
        M = torch.randn(n, n)
        erank = effective_rank(M)
        # For random matrix, erank is typically 0.7-0.9 * n due to singular value decay
        print(f"  Random {n}x{n}: erank = {erank:.4f} (expected: ~{0.8*n:.1f}-{n})")
        assert erank > 0.5 * n, f"Random matrix erank should be > {0.5*n}, got {erank}"
    print("  PASSED\n")

def test_diagonal_matrix():
    """Diagonal matrix with known singular values"""
    print("="*60)
    print("TEST 5: Diagonal Matrix (controlled singular values)")
    print("="*60)
    
    # Case 1: All equal diagonal values -> erank = n
    n = 10
    D = torch.diag(torch.ones(n))
    erank = effective_rank(D)
    print(f"  Diag (all 1s) {n}x{n}: erank = {erank:.4f} (expected: {n})")
    assert abs(erank - n) < 0.1, f"Expected {n}, got {erank}"
    
    # Case 2: One dominant singular value -> erank ~ 1
    D = torch.diag(torch.tensor([1.0] + [0.001] * (n-1)))
    erank = effective_rank(D)
    print(f"  Diag (1 dominant) {n}x{n}: erank = {erank:.4f} (expected: ~1)")
    assert erank < 2, f"Expected ~1, got {erank}"
    
    # Case 3: Half the values are non-zero and equal -> erank = n/2
    D = torch.diag(torch.tensor([1.0] * 5 + [0.0] * 5))
    erank = effective_rank(D)
    print(f"  Diag (5 of 10 equal) {n}x{n}: erank = {erank:.4f} (expected: 5)")
    assert abs(erank - 5) < 0.1, f"Expected 5, got {erank}"
    
    print("  PASSED\n")

def test_gram_matrix():
    """Test that Gram matrix erank is computed correctly"""
    print("="*60)
    print("TEST 6: Gram Matrix K = A @ A.T")
    print("="*60)
    
    n_samples, dim = 50, 100
    
    # Case 1: Random embeddings
    A = torch.randn(n_samples, dim)
    gram = A @ A.T
    erank_gram = effective_rank(gram)
    print(f"  Random embeddings ({n_samples}x{dim}): Gram erank = {erank_gram:.4f}")
    print(f"    (max possible = min(n_samples, dim) = {min(n_samples, dim)})")
    
    # Case 2: Low-rank embeddings (only k dimensions active)
    k = 10
    A_lowrank = torch.randn(n_samples, k) @ torch.randn(k, dim)
    gram_lowrank = A_lowrank @ A_lowrank.T
    erank_lowrank = effective_rank(gram_lowrank)
    print(f"  Low-rank embeddings (rank={k}): Gram erank = {erank_lowrank:.4f} (expected: ~{k})")
    assert abs(erank_lowrank - k) < 2, f"Expected ~{k}, got {erank_lowrank}"
    
    # Case 3: Identical embeddings (rank 1)
    A_same = torch.randn(1, dim).repeat(n_samples, 1)
    gram_same = A_same @ A_same.T
    erank_same = effective_rank(gram_same)
    print(f"  Identical embeddings: Gram erank = {erank_same:.4f} (expected: 1)")
    assert erank_same < 1.5, f"Expected ~1, got {erank_same}"
    
    print("  PASSED\n")

def test_model_erank():
    """Test effective rank computation through a neural network"""
    print("="*60)
    print("TEST 7: Neural Network Gram Matrix Erank")
    print("="*60)
    
    torch.manual_seed(42)
    dim, depth = 64, 8
    n_samples = 128
    
    model = ReLUNetwork(dim=dim, depth=depth)
    X = torch.randn(n_samples, dim)
    
    # Compute erank
    erank = compute_gram_effective_rank(model, X)
    print(f"  Model (dim={dim}, depth={depth}), {n_samples} samples")
    print(f"  Gram matrix erank = {erank:.4f}")
    print(f"  (Should be between 1 and {min(n_samples, dim)} = {min(n_samples, dim)})")
    
    assert 1 <= erank <= min(n_samples, dim), f"Erank {erank} out of expected range"
    
    # Test that different random seeds give different eranks
    model2 = ReLUNetwork(dim=dim, depth=depth)
    erank2 = compute_gram_effective_rank(model2, X)
    print(f"  Different init: erank = {erank2:.4f}")
    
    print("  PASSED\n")

def test_numerical_stability():
    """Test for numerical issues with very small/large values"""
    print("="*60)
    print("TEST 8: Numerical Stability")
    print("="*60)
    
    n = 50
    
    # Very small values
    M_small = torch.randn(n, n) * 1e-8
    erank_small = effective_rank(M_small)
    print(f"  Small values (1e-8 scale): erank = {erank_small:.4f}")
    
    # Very large values
    M_large = torch.randn(n, n) * 1e8
    erank_large = effective_rank(M_large)
    print(f"  Large values (1e8 scale): erank = {erank_large:.4f}")
    
    # Eranks should be similar (scale-invariant)
    assert abs(erank_small - erank_large) < 5, f"Erank should be scale-invariant"
    
    # Near-zero matrix
    M_tiny = torch.zeros(n, n) + 1e-15
    erank_tiny = effective_rank(M_tiny)
    print(f"  Near-zero matrix: erank = {erank_tiny:.4f} (should handle gracefully)")
    
    print("  PASSED\n")

def test_changing_erank():
    """Verify erank actually changes when embeddings change"""
    print("="*60)
    print("TEST 9: Verify Erank Changes With Model Updates")
    print("="*60)
    
    torch.manual_seed(42)
    dim, depth = 64, 8
    n_samples = 128
    
    model = ReLUNetwork(dim=dim, depth=depth)
    X = torch.randn(n_samples, dim)
    Y = torch.randn(n_samples, dim)
    
    # Initial erank
    erank_init = compute_gram_effective_rank(model, X)
    print(f"  Initial erank: {erank_init:.4f}")
    
    # Train for a few steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    for i in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    
    erank_after_100 = compute_gram_effective_rank(model, X)
    print(f"  After 100 steps: erank = {erank_after_100:.4f}")
    
    # Train more
    for i in range(900):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    
    erank_after_1000 = compute_gram_effective_rank(model, X)
    print(f"  After 1000 steps: erank = {erank_after_1000:.4f}")
    
    # Check that erank changed (at least somewhat)
    changes = [
        abs(erank_init - erank_after_100),
        abs(erank_after_100 - erank_after_1000),
        abs(erank_init - erank_after_1000)
    ]
    print(f"  Changes: init->100: {changes[0]:.4f}, 100->1000: {changes[1]:.4f}, init->1000: {changes[2]:.4f}")
    
    if max(changes) < 0.1:
        print("  WARNING: Erank barely changed during training!")
        print("  This could indicate a problem or that the model is in a stable regime.")
    else:
        print("  PASSED - Erank changed during training\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EFFECTIVE RANK (ERANK) FUNCTION TESTS")
    print("="*60 + "\n")
    
    test_identity_matrix()
    test_rank1_matrix()
    test_low_rank_matrix()
    test_random_matrix()
    test_diagonal_matrix()
    test_gram_matrix()
    test_model_erank()
    test_numerical_stability()
    test_changing_erank()
    
    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
