import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set directories
figs_dir = "../figs/general"
csv_file = "../results/general/poisson_error_norms.csv"

# Ensure output directory exists
os.makedirs(figs_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract data
mesh_sizes = np.array([3, 4, 6, 8, 12, 20, 24, 48, 96], dtype=float)
h_values = 1 / mesh_sizes  # Compute h values (element size)
L2_norms = df["L2 Norm"].values
H1_norms = df["H1 Norm"].values
Relative_L2_norms = df["Relative L2 Norm"].values

# Compute min/max h values for reference lines
h_min = np.min(h_values)
h_max = np.max(h_values)
h_ref = np.array([h_max, h_min])  # Use first and last h values

# Select a reference index for scaling reference slopes
ref_index = len(df) // 2
ref_h = h_values[ref_index]
ref_L2 = L2_norms[ref_index]
ref_H1 = H1_norms[ref_index]
ref_Relative_L2 = Relative_L2_norms[ref_index]

# Define reference lines for O(h^2) and O(h)
o_h2 = ref_Relative_L2 * (h_ref / ref_h) ** 2  # Quadratic convergence
o_h1 = ref_H1 * (h_ref / ref_h)  # Linear convergence

# --- Plot 1: Relative L2 Norm vs. h with O(h^2) Reference ---
plt.figure(figsize=(8, 6))
plt.loglog(h_values, Relative_L2_norms, marker='o', linestyle='-', label="Relative L2 Norm")

# Plot reference lines
plt.loglog(h_ref, o_h2, 'k--', label=r"$O(h^2)$ (Reference Slope)")

plt.xlabel("h (Element Size)")
plt.ylabel("Relative L2 Norm")
plt.title("Relative L2 Norm vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "relative_L2_vs_h.png"), dpi=300)
plt.show()

# --- Plot 2: Convergence of L2 and H1 Norms with O(h^2) and O(h) References ---
plt.figure(figsize=(8, 6))
plt.loglog(h_values, L2_norms, marker='s', linestyle='-', label="L2 Norm")
plt.loglog(h_values, H1_norms, marker='^', linestyle='-', label="H1 Norm")

# Plot reference lines
slope_L2_ref = ref_L2 * (h_ref / ref_h) ** 2  # Second-order slope for L2
slope_H1_ref = ref_H1 * (h_ref / ref_h)  # First-order slope for H1

plt.loglog(h_ref, slope_L2_ref, 'k--', label=r"$O(h^2)$ (Reference Slope for L2)")
plt.loglog(h_ref, slope_H1_ref, 'r--', label=r"$O(h)$ (Reference Slope for H1)")

plt.xlabel("h (Element Size)")
plt.ylabel("Error Norms")
plt.title("Convergence Plot: L2 and H1 Norms vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "convergence.png"), dpi=300)
plt.show()
