import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load subdomain results from CSV
results_filename = "../results/subdomains/subdomain_error_norms.csv"
df = pd.read_csv(results_filename)

# Ensure numerical types
df[["mesh_x", "mesh_y", "sub_x0", "sub_x1", "sub_y0", "sub_y1", "L2_norm", "H1_norm", "Relative_L2_norm"]] = \
    df[["mesh_x", "mesh_y", "sub_x0", "sub_x1", "sub_y0", "sub_y1", "L2_norm", "H1_norm", "Relative_L2_norm"]].astype(float)

# Compute h values (element size)
df["h"] = 1 / df["mesh_x"]  # Assuming square elements with uniform spacing

# Create directory for figures
figs_dir = "../figs/general/subdomains"
os.makedirs(figs_dir, exist_ok=True)

# Plot Relative L2 Norm vs. h (All Subdomains on All Meshes)
plt.figure(figsize=(8, 6))
for _, row in df.iterrows():
    plt.loglog(row["h"], row["Relative_L2_norm"], marker='o', linestyle='-', label=f"Subdomain {row['sub_x0']:.2f}-{row['sub_x1']:.2f}, {row['sub_y0']:.2f}-{row['sub_y1']:.2f}")

# Add reference O(h^2) slope
h_ref = np.array([df["h"].min(), df["h"].max()])
slope_ref = df["Relative_L2_norm"].max() * (h_ref / df["h"].max())**2
plt.loglog(h_ref, slope_ref, 'k--', label=r"$O(h^2)$ (Reference Slope)")

plt.xlabel("h (Element Size)")
plt.ylabel("Relative L2 Norm")
plt.title("Subdomain Convergence: Relative L2 Norm vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "subdomain_relative_L2_vs_h.png"), dpi=300)
plt.show()

# Plot Convergence of L2 and H1 Norms vs. h (All Subdomains on All Meshes)
plt.figure(figsize=(8, 6))
for _, row in df.iterrows():
    plt.loglog(row["h"], row["L2_norm"], marker='s', linestyle='-', label=f"L2 Norm {row['sub_x0']:.2f}-{row['sub_x1']:.2f}, {row['sub_y0']:.2f}-{row['sub_y1']:.2f}")
    plt.loglog(row["h"], row["H1_norm"], marker='^', linestyle='-', label=f"H1 Norm {row['sub_x0']:.2f}-{row['sub_x1']:.2f}, {row['sub_y0']:.2f}-{row['sub_y1']:.2f}")

# Add reference O(h^2) slope
slope_L2_ref = df["L2_norm"].max() * (h_ref / df["h"].max())**2
slope_H1_ref = df["H1_norm"].max() * (h_ref / df["h"].max())**2
plt.loglog(h_ref, slope_L2_ref, 'k--', label=r"$O(h^2)$ (Reference Slope for L2)")
plt.loglog(h_ref, slope_H1_ref, 'r--', label=r"$O(h^2)$ (Reference Slope for H1)")

plt.xlabel("h (Element Size)")
plt.ylabel("Error Norms")
plt.title("Subdomain Convergence: L2 and H1 Norms vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "subdomain_convergence.png"), dpi=300)
plt.show()
