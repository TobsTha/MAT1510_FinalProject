"""
Continue training from a checkpoint.
Use this to extend training beyond the initial run.

Example: If you trained for 100k epochs and want to add 50k more,
run this script with ADDITIONAL_EPOCHS = 50000
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
# Add parent directory to path to import DD_R32_D48_training
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# Import from main script (parent directory)
from DD_R32_D48_training import (
    ReLUNetwork,
    create_task_matrix,
    generate_data,
    train_with_tracking,
    save_results,
    save_checkpoint,
    compute_gram_effective_rank,
    device,
)


# Setup logging
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")  # Append mode

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(os.path.join(SCRIPT_DIR, "DD_R32_D48_continue_training.log"))


if __name__ == "__main__":
    # Configuration - use absolute paths
    CHECKPOINT_DIR = os.path.join(PARENT_DIR, "checkpoints")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "DD_R32_D48_checkpoint.pt")
    ADDITIONAL_EPOCHS = 100000  # How many more epochs to train

    # Must match original experiment
    RANK = 32
    DEPTH = 48
    dim, n_samples, seed = 64, 128, 42

    print("\n" + "=" * 70)
    print("CONTINUING TRAINING FROM CHECKPOINT")
    print("=" * 70)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    start_epoch = checkpoint["epoch"]
    previous_loss_history = checkpoint.get("loss_history", [])
    previous_erank_history = checkpoint.get("erank_history", [])

    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"Adding {ADDITIONAL_EPOCHS} more epochs")
    print(f"Will train to epoch {start_epoch + ADDITIONAL_EPOCHS}")

    # Recreate task (must use same seed!)
    W = create_task_matrix(dim, RANK, seed=seed)
    X, Y = generate_data(W, n_samples, seed=seed + 1)
    baseline_loss = (Y**2).mean().item()

    # Recreate model and load weights
    model = ReLUNetwork(dim=dim, depth=DEPTH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Get current loss and effective rank
    criterion = nn.MSELoss()
    with torch.no_grad():
        current_loss = criterion(model(X), Y).item()
    current_erank = compute_gram_effective_rank(model, X)

    print(f"Current loss at epoch {start_epoch}: {current_loss:.6f}")
    print(f"Current effective rank at epoch {start_epoch}: {current_erank:.2f}")

    # If no previous erank history, start with current erank
    if not previous_erank_history:
        previous_erank_history = [(start_epoch, current_erank)]
        print(f"Starting effective rank tracking from epoch {start_epoch}")

    # Continue training
    start_time = time.time()

    # Use same LR as final LR from checkpoint (or specify new one)
    lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
    print(f"Continuing with LR: {lr}")

    loss_history, erank_history, final_loss = train_with_tracking(
        model,
        X,
        Y,
        epochs=ADDITIONAL_EPOCHS,
        lr=lr,
        log_every=100,
        print_every=5000,
        checkpoint_every=10000,
        checkpoint_path=CHECKPOINT_PATH,
        use_scheduler=True,
        start_epoch=start_epoch,
        erank_every=5000,
    )

    total_time = time.time() - start_time
    final_epoch = start_epoch + ADDITIONAL_EPOCHS

    # Combine histories
    combined_loss_history = previous_loss_history + loss_history
    combined_erank_history = previous_erank_history + erank_history

    # Get final effective rank
    final_erank = combined_erank_history[-1][1] if combined_erank_history else 0
    initial_erank = combined_erank_history[0][1] if combined_erank_history else 0

    # Save updated results
    results = {
        "rank": RANK,
        "depth": DEPTH,
        "dim": dim,
        "n_samples": n_samples,
        "epochs": final_epoch,
        "lr": lr,
        "seed": seed,
        "baseline_loss": baseline_loss,
        "best_final_loss": final_loss,
        "best_lr": lr,
        "total_time_seconds": total_time,
        "initial_erank": initial_erank,
        "final_erank": final_erank,
        "erank_history": combined_erank_history,
        "experiments": [
            {"lr": lr, "final_loss": final_loss, "loss_history": combined_loss_history}
        ],
    }

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Trained from epoch {start_epoch} to {final_epoch}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Baseline loss: {baseline_loss:.2f}")
    print(
        f"Improvement: {baseline_loss / final_loss:.1f}x" if final_loss > 0 else "N/A"
    )
    print(f"Initial effective rank: {initial_erank:.2f}")
    print(f"Final effective rank: {final_erank:.2f}")
    print(f"Additional training time: {total_time / 60:.1f} minutes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = os.path.join(PARENT_DIR, "data")
    save_results(
        results, os.path.join(DATA_DIR, f"DD_R32_D48_results_{timestamp}.json")
    )
    save_results(results, os.path.join(DATA_DIR, "DD_R32_D48_results.json"))

    print(f"\nUpdated checkpoint saved to: {CHECKPOINT_PATH}")
    print("Run 'DD_R32_D48_erank_plot.ipynb' to visualize effective rank evolution.")
