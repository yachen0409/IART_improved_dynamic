import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data
k = 7
# Continuous curve data
d_range = np.linspace(0, 1, 500)
w_range = np.exp(-k * d_range)

# Specific sample points you requested
sample_d = np.array([0.01, 0.2, 0.4, 0.6, 0.8, 0.99])
sample_w = np.exp(-k * sample_d)

# 2. Create Plot
plt.figure(figsize=(10, 6), dpi=150)

# Plot the main curve
plt.plot(d_range, w_range, color='#0066cc', linewidth=2.5, label=r'Weight Function: $w = e^{-7 \cdot d}$')
plt.fill_between(d_range, w_range, color='#e6f3ff', alpha=0.5)

# Plot the specific sample points
plt.scatter(sample_d, sample_w, color='red', s=80, zorder=5, edgecolor='black')

# 3. Annotate the points
for d_val, w_val in zip(sample_d, sample_w):
    # Format the label text
    label_text = f"d={d_val}\nw={w_val:.3f}"
    
    # Adjust text position based on where the point is
    if d_val == 0.2:
        xytext = (10, 20) # Shift up-right
        arrow_color = 'black'
    elif d_val == 0.4:
        xytext = (10, 20) 
        arrow_color = 'black'
    else:
        # For very low values, shift text up so it doesn't overlap axis
        xytext = (0, 30) 
        arrow_color = 'gray'

    plt.annotate(label_text, 
                 xy=(d_val, w_val), 
                 xytext=xytext, 
                 textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=arrow_color))

# 4. Styling
plt.title(r'Impact of Dissimilarity on Weight ($k=7$)', fontsize=14, fontweight='bold')
plt.xlabel('Dissimilarity ($d$)', fontsize=12)
plt.ylabel('Weight ($w$)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

# Set limits
plt.xlim(0, 1.1)
plt.ylim(-0.05, 1.1)

plt.tight_layout()


# --- ADDED LINES START HERE ---
try:
    # Get the name of the current script file (without extension)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    # Fallback if running in Jupyter/Colab where __file__ is not defined
    script_name = "dissimilarity_weight_curve"

output_filename = f"{script_name}.png"

# Save the figure BEFORE showing it
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ Figure successfully saved as: {output_filename}")
# --- ADDED LINES END HERE ---
