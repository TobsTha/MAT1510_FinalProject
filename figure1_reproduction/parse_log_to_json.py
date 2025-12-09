"""
Parse the training log file to extract results and save as JSON.
Use this if the experiment completed but crashed before saving.
"""

import re
import json

log_file = 'figure1_training.log'
output_file = 'figure1_results.json'

# Initialize results structure
results = {
    'depths': [2, 4, 8, 16, 32, 48, 64],
    'ranks': [1, 4, 16, 32, 64],
    'base_lr': 0.1,
    'lr_multipliers': [0.1, 0.25, 0.5, 1.0, 2.0],
    'n_task_seeds': 1,
    'n_init_seeds': 1,
    'epochs': 4000,
    'dim': 64,
    'n_samples': 128,
    'losses': {},
    'best_lrs': {},
    'total_time_seconds': 70.8 * 60
}

# Parse log file
with open(log_file, 'r') as f:
    content = f.read()

# Pattern to match: "Depth  2: loss = 0.000048 +/- 0.000000, best_lr ~ 0.1414"
pattern = r'Depth\s+(\d+): loss = ([\d.]+) \+/- [\d.]+, best_lr ~ ([\d.]+)'

current_rank = None
for line in content.split('\n'):
    # Check for rank header
    rank_match = re.search(r'Task Rank: (\d+)', line)
    if rank_match:
        current_rank = int(rank_match.group(1))
        results['losses'][str(current_rank)] = {}
        results['best_lrs'][str(current_rank)] = {}
        continue
    
    # Check for depth results
    depth_match = re.search(pattern, line)
    if depth_match and current_rank is not None:
        depth = int(depth_match.group(1))
        loss = float(depth_match.group(2))
        best_lr = float(depth_match.group(3))
        
        results['losses'][str(current_rank)][str(depth)] = [loss]
        results['best_lrs'][str(current_rank)][str(depth)] = [best_lr]

# Save to JSON
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print(f"\nExtracted data for {len(results['ranks'])} ranks x {len(results['depths'])} depths")

# Print summary
print("\nLoss Summary:")
print(f"{'Rank':<8}", end='')
for d in results['depths']:
    print(f"D={d:<7}", end='')
print()
print("-" * 64)

for rank in results['ranks']:
    print(f"{rank:<8}", end='')
    for depth in results['depths']:
        loss = results['losses'][str(rank)][str(depth)][0]
        print(f"{loss:<8.4f}", end='')
    print()
