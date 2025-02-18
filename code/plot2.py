import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load subdomain results from CSV
results_filename = "../results/subdomains/subdomain_error_norms.csv"
df = pd.read_csv(results_filename)

# Compute h values (element size) assuming uniform mesh refinement
df["h"] = 1 / df["mesh_x"]

# Group by subdomains to maintain consistency across different meshes
subdomain_groups = df.groupby(["sub_x0", "sub_x1", "sub_y0", "sub_y1"])

# Create figures directory
figs_dir = "../figs/subdomains"
os.makedirs(figs_dir, exist_ok=True)

### **Plot 1: Relative L2 Norm vs. h (All meshes on one plot, grouped by subdomains)**
plt.figure(figsize=(10, 7))

for (sub_x0, sub_x1, sub_y0, sub_y1), sub_df in subdomain_groups:
    label = f"Sub ({sub_x0:.2f}, {sub_x1:.2f}, {sub_y0:.2f}, {sub_y1:.2f})"
    plt.loglog(sub_df["h"], sub_df["Relative_L2_norm"], marker='o', linestyle='-', label=label)

# Add reference O(h^2) slope
h_ref = np.array([df["h"].min(), df["h"].max()])
slope_ref = df["Relative_L2_norm"].max() * (h_ref / df["h"].max())**2
plt.loglog(h_ref, slope_ref, 'k--', label=r"$O(h^2)$ (Reference Slope)")

plt.xlabel("h (Element Size)")
plt.ylabel("Relative L2 Norm")
plt.title("Subdomain Convergence: Relative L2 Norm vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend(fontsize=8, loc="best")
plt.savefig(os.path.join(figs_dir, "subdomain_relative_L2_vs_h.png"), dpi=300)
plt.show()

### **Plot 2: L2 and H1 Norm Convergence vs. h (All meshes on one plot, grouped by subdomains)**
plt.figure(figsize=(10, 7))

for (sub_x0, sub_x1, sub_y0, sub_y1), sub_df in subdomain_groups:
    label = f"Sub ({sub_x0:.2f}, {sub_x1:.2f}, {sub_y0:.2f}, {sub_y1:.2f})"
    plt.loglog(sub_df["h"], sub_df["L2_norm"], marker='s', linestyle='-', label=f"{label} (L2)")
    plt.loglog(sub_df["h"], sub_df["H1_norm"], marker='^', linestyle='-', label=f"{label} (H1)")

# Add reference O(h^2) for L2 and O(h) for H1 slopes
slope_L2_ref = df["L2_norm"].max() * (h_ref / df["h"].max())**2
slope_H1_ref = df["H1_norm"].max() * (h_ref / df["h"].max())**1

plt.loglog(h_ref, slope_L2_ref, 'k--', label=r"$O(h^2)$ (Reference Slope for L2)")
plt.loglog(h_ref, slope_H1_ref, 'r--', label=r"$O(h)$ (Reference Slope for H1)")

plt.xlabel("h (Element Size)")
plt.ylabel("Error Norms")
plt.title("Subdomain Convergence: L2 and H1 Norms vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend(fontsize=8, loc="best")
plt.savefig(os.path.join(figs_dir, "subdomain_convergence.png"), dpi=300)
plt.show()
