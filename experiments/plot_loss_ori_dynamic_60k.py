import re
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# List your files here. 
# BASELINE: The two files must be in chronological order.
baseline_files = [
    "IART_REDS_N5_60K/train_IART_REDS_N5_60K_20251123_123733.log",
    "IART_REDS_N5_60K/train_IART_REDS_N5_60K_20251125_155521.log"
]

# IMPROVED: The single log file.
improved_files = [
    "IART_REDS_N5_60K_Improved_Dynamic/train_IART_REDS_N5_60K_Improved_Dynamic_20251125_230122.log"
]
SMOOTHING_WINDOW = 100 

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
        count = 0
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # CHECK 1: Look for "l_pix" OR "l_g_pix"
                    if "l_pix:" in line or "l_g_pix:" in line:
                        
                        # CHECK 2: Handle commas in numbers (e.g., "59,975")
                        # Regex explanation:
                        # iter:\s*([\d,]+)  -> Finds "iter: " followed by digits AND commas
                        # l_g?_?pix:\s*...  -> Finds "l_pix:" or "l_g_pix:"
                        iter_match = re.search(r'iter:\s*([\d,]+)', line)
                        loss_match = re.search(r'l_g?_?pix:\s*([\d\.e-]+)', line)
                        
                        if iter_match and loss_match:
                            # Remove comma before converting to int
                            iter_str = iter_match.group(1).replace(',', '')
                            it = int(iter_str)
                            loss = float(loss_match.group(1))
                            
                            iters.append(it)
                            losses.append(loss)
                            count += 1
        except Exception as e:
            print(f"Error reading file: {e}")

        print(f"   -> Found {count} data points.")
            
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
    print("\n❌ CRITICAL ERROR: No data found. Check filenames.")
    exit()

# Smooth Data
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

# Plotting
plt.figure(figsize=(10, 6), dpi=150)

if base_iters_smooth:
    plt.plot(base_iters_smooth, base_losses_smooth, 
             color='#e74c3c', label='Baseline (Original)', linewidth=1.5, alpha=0.6)

if imp_iters_smooth:
    plt.plot(imp_iters_smooth, imp_losses_smooth, 
             color='#2980b9', label='Ours (Dynamic Weighting)', linewidth=2.0)

plt.title("Training Loss Comparison (0 - 60k Iterations)", fontsize=14, fontweight='bold')
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Pixel Loss (l_pix)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Add annotation if data exists
if len(imp_iters_smooth) > 0:
    last_x = imp_iters_smooth[-1]
    last_y = imp_losses_smooth[-1]
    plt.annotate(f'Final Loss: {last_y:.4f}', 
                 xy=(last_x, last_y), 
                 xytext=(last_x - 15000, last_y + 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
output_filename = "loss_comparison_final.png"
plt.savefig(output_filename)
print(f"\n✅ Success! Chart saved as {output_filename}")
plt.show()