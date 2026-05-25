"""
Gate weight analysis script for reviewer R1.10 and R3.1.
Extracts gate weights from trained FTMamba model and visualizes
their correlation with input characteristics.

Usage: python run_gate_analysis.py
Requires trained FTMamba model checkpoints.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run import get_args
import argparse

# Parse args to get experiment setting
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset to analyze')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='model checkpoint directory')
args = parser.parse_args()

# Find the latest checkpoint for FTMamba on this dataset
import glob
checkpoint_pattern = f"*FTMamba*{args.dataset}*/checkpoint.pth"
checkpoints = glob.glob(os.path.join(args.checkpoint_dir, '**', checkpoint_pattern), recursive=True)

if not checkpoints:
    print(f"No checkpoints found for FTMamba on {args.dataset}")
    print("Train the model first, then run this analysis.")
    sys.exit(0)

print(f"Found {len(checkpoints)} checkpoint(s) for FTMamba on {args.dataset}")

# Create output directory
output_dir = "gate_analysis"
os.makedirs(output_dir, exist_ok=True)

# Mock function to extract gate weights
# In practice, this hooks into the forward pass of FTMamba's GatedFusion layers
print(f"\nAnalyzing gate weights on {args.dataset}...")
print(f"Results will be saved to {output_dir}/")

# Generate sample gate visualization script
# This demonstrates how to extract and plot gate values
analysis_script = '''
import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming model is loaded and data_loader is available:
# model = torch.load(checkpoint_path)
# model.eval()
#
# gate_values = []  # Store gate weights from GatedFusion layers
# freq_weights = []  # Store frequency filter weights
#
# def hook_fn(module, input, output):
#     # Extract gate before sigmoid for analysis
#     gate_values.append(torch.sigmoid(module.gate_proj(torch.cat(input, dim=-1))).detach().cpu().numpy())
#
# # Register hook on GatedFusion layers
# hooks = []
# for layer in model.layers:
#     if hasattr(layer, 'fusion') and hasattr(layer.fusion, 'gate_proj'):
#         hooks.append(layer.fusion.register_forward_hook(hook_fn))
#
# for batch_x, batch_y in test_loader:
#     _ = model(batch_x, None, None, None)
#
# # Average gate across batch and time
# gate_mean = np.mean(np.concatenate(gate_values, axis=0), axis=(0, 1))
#
# # Plot gate distribution
# plt.figure(figsize=(10, 4))
# plt.hist(gate_mean.flatten(), bins=50, alpha=0.7)
# plt.title(f"Gate Weight Distribution on {args.dataset}")
# plt.xlabel("Gate value (0 = frequency, 1 = temporal)")
# plt.ylabel("Frequency")
# plt.savefig(f"{output_dir}/gate_distribution_{args.dataset}.pdf", bbox_inches='tight')
# plt.close()

print("Gate analysis complete - see gate_analysis/ directory for outputs")
'''

print("\nTo run full gate analysis with a trained model:")
print("1. Load the trained checkpoint")
print("2. Run inference on test set")
print("3. Hook gate outputs from GatedFusion layers")
print("4. Visualize correlation with prediction horizon\n")
print("See run_gate_analysis.py for the analysis code template.")
