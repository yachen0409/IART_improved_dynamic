import re
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# List your files here. 
# BASELINE: The two files must be in chronological order.
baseline_files = [
    "IART_REDS_N5_15K/train_IART_REDS_N5_15K_20251121_170730.log",
    "IART_REDS_N5_15K/train_IART_REDS_N5_15K_20251121_202949.log"
]

# IMPROVED: The single log file.
improved_files = [
    "IART_REDS_N5_15K_Improved/train_IART_REDS_N5_15K_Improved_20251122_111025.log",
    "IART_REDS_N5_15K_Improved/train_IART_REDS_N5_15K_Improved_20251122_152840.log"
]

SMOOTHING_WINDOW = 50 

# -----------------------------------------

def parse_logs(file_list):
    iters = []
    losses = []
    
    print(f"--- Parsing Group ---")
    for filename in file_list:
        filename = filename.strip()
        if not os.path.exists(filename):
            print(f"❌ ERROR: File not found: {filename}")
            continue
            
        print(f"Reading: {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "l_pix:" in line or "l_g_pix:" in line:
                        iter_match = re.search(r'iter:\s*([\d,]+)', line)
                        loss_match = re.search(r'l_g?_?pix:\s*([\d\.e-]+)', line)
                        
                        if iter_match and loss_match:
                            iter_str = iter_match.group(1).replace(',', '')
                            it = int(iter_str)
                            loss = float(loss_match.group(1))
                            iters.append(it)
                            losses.append(loss)
        except Exception as e:
            print(f"Error reading file: {e}")
            
    return iters, losses

def smooth_data(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- MAIN PROCESSING ---

print("Processing Baseline Logs...")
base_iters, base_losses = parse_logs(baseline_files)

print("\nProcessing Improved Logs...")
imp_iters, imp_losses = parse_logs(improved_files)

if not base_iters and not imp_iters:
    print("\n❌ CRITICAL ERROR: No data found.")
    exit()

# --- ⚡ TRUNCATION LOGIC (NEW) ---
# If both exist, find the minimum end point and cut the longer one.
if base_iters and imp_iters:
    min_end = min(base_iters[-1], imp_iters[-1])
    print(f"\n✂️ Truncating graph to common iteration: {min_end}")

    # Truncate Baseline
    base_indices = [i for i, x in enumerate(base_iters) if x <= min_end]
    base_iters = [base_iters[i] for i in base_indices]
    base_losses = [base_losses[i] for i in base_indices]

    # Truncate Improved
    imp_indices = [i for i, x in enumerate(imp_iters) if x <= min_end]
    imp_iters = [imp_iters[i] for i in imp_indices]
    imp_losses = [imp_losses[i] for i in imp_indices]

# --- SMOOTHING ---
if base_losses:
    base_losses_smooth = smooth_data(base_losses, SMOOTHING_WINDOW)
    base_iters_smooth = base_iters[len(base_iters)-len(base_losses_smooth):]
else:
    base_iters_smooth, base_losses_smooth = [], []

if imp_losses:
    imp_losses_smooth = smooth_data(imp_losses, SMOOTHING_WINDOW)
    imp_iters_smooth = imp_iters[len(imp_iters)-len(imp_losses_smooth):]
else:
    imp_iters_smooth, imp_losses_smooth = [], []

# --- PLOTTING ---
plt.figure(figsize=(10, 6), dpi=150)

if base_iters_smooth:
    plt.plot(base_iters_smooth, base_losses_smooth, 
             color='#e74c3c', label='Baseline', linewidth=1.5, alpha=0.6)

if imp_iters_smooth:
    plt.plot(imp_iters_smooth, imp_losses_smooth, 
             color='#2980b9', label='Ours (Fixed)', linewidth=2.0)

# Title and Labels
max_iter = base_iters[-1] if base_iters else imp_iters[-1]
plt.title(f"Training Loss Comparison (0 - {max_iter} Iterations)", fontsize=14, fontweight='bold')
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("L_total", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Annotation
if len(imp_iters_smooth) > 0:
    last_x = imp_iters_smooth[-1]
    last_y = imp_losses_smooth[-1]
    x_offset = max_iter * 0.25 
    

plt.tight_layout()
output_filename = "loss_comparison_ori_fixed_10k.png"
plt.savefig(output_filename)
print(f"\n✅ Success! Chart saved as {output_filename}")
plt.show()