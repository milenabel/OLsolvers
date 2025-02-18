# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the results file path again
results_filename = "../results/subdomains/subdomain_error_norms.csv"

# Reload the data
df = pd.read_csv(results_filename)

# Extract unique mesh sizes
mesh_sizes = df[["mesh_x", "mesh_y"]].drop_duplicates().values

# Create figures directory
figs_dir = "../figs/subdomains"
os.makedirs(figs_dir, exist_ok=True)

# Iterate over each mesh size and plot subdomain convergence and relative L2 error
for mesh_x, mesh_y in mesh_sizes:
    mesh_df = df[(df["mesh_x"] == mesh_x) & (df["mesh_y"] == mesh_y)]
    
    # Compute h values (element size)
    mesh_df["h"] = 1 / (mesh_df["sub_x1"] - mesh_df["sub_x0"])

    # Plot convergence of L2 and H1 norms
    plt.figure(figsize=(8, 6))
    for sub_x0, sub_x1, sub_y0, sub_y1 in zip(mesh_df["sub_x0"], mesh_df["sub_x1"], mesh_df["sub_y0"], mesh_df["sub_y1"]):
        subdomain_label = f"({sub_x0:.2f}, {sub_x1:.2f}, {sub_y0:.2f}, {sub_y1:.2f})"
        subdomain_df = mesh_df[
            (mesh_df["sub_x0"] == sub_x0) & (mesh_df["sub_x1"] == sub_x1) &
            (mesh_df["sub_y0"] == sub_y0) & (mesh_df["sub_y1"] == sub_y1)
        ]
        plt.loglog(subdomain_df["h"], subdomain_df["L2_norm"], marker='o', linestyle='-', label=f"L2 {subdomain_label}")
        plt.loglog(subdomain_df["h"], subdomain_df["H1_norm"], marker='^', linestyle='-', label=f"H1 {subdomain_label}")

    plt.xlabel("h (Element Size)")
    plt.ylabel("Error Norms")
    plt.title(f"Convergence Plot: L2 and H1 Norms vs. h for Mesh ({mesh_x}, {mesh_y})")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(figs_dir, f"convergence_{mesh_x}x{mesh_y}.png"), dpi=300)
    plt.show()

    # Plot relative L2 error for each subdomain
    plt.figure(figsize=(8, 6))
    for sub_x0, sub_x1, sub_y0, sub_y1 in zip(mesh_df["sub_x0"], mesh_df["sub_x1"], mesh_df["sub_y0"], mesh_df["sub_y1"]):
        subdomain_label = f"({sub_x0:.2f}, {sub_x1:.2f}, {sub_y0:.2f}, {sub_y1:.2f})"
        subdomain_df = mesh_df[
            (mesh_df["sub_x0"] == sub_x0) & (mesh_df["sub_x1"] == sub_x1) &
            (mesh_df["sub_y0"] == sub_y0) & (mesh_df["sub_y1"] == sub_y1)
        ]
        plt.loglog(subdomain_df["h"], subdomain_df["Relative_L2_norm"], marker='s', linestyle='-', label=f"Rel L2 {subdomain_label}")

    plt.xlabel("h (Element Size)")
    plt.ylabel("Relative L2 Norm")
    plt.title(f"Relative L2 Error vs. h for Mesh ({mesh_x}, {mesh_y})")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(figs_dir, f"relative_L2_{mesh_x}x{mesh_y}.png"), dpi=300)
    plt.show()
