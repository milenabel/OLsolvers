import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set directories
figs_dir = "../figs/subdomains"
csv_file = "../results/subdomains/subdomain_error_norms.csv"

# Ensure output directory exists
os.makedirs(figs_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract unique global mesh sizes
global_mesh_sizes = df[['global_mesh_x', 'global_mesh_y']].drop_duplicates().values
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colors for multiple plots

# --- Plot 1: Relative L2 Norm vs. h with O(h^2) Reference ---
plt.figure(figsize=(8, 6))

for i, (Nx, Ny) in enumerate(global_mesh_sizes):
    subset = df[(df['global_mesh_x'] == Nx) & (df['global_mesh_y'] == Ny)]
    h_subdomain = 1 / subset['sub_mesh_x']  # Compute h values for subdomain mesh
    plt.loglog(h_subdomain, subset['Relative_L2_norm'], marker='o', linestyle='-', color=colors[i % len(colors)],
               label=f"Global Mesh: {Nx}x{Ny}")

# Add reference line for O(h^2) (quadratic convergence)
h_ref = np.array([min(h_subdomain), max(h_subdomain)])
o_h2 = 10 * h_ref**2  # Adjust scaling to fit within the plot range
o_h1 = 10 * h_ref  # Linear convergence
plt.loglog(h_ref, o_h2, 'k--', label=r"$O(h^2)$")  # Dashed black line
plt.loglog(h_ref, o_h1, 'k-.', label=r"$O(h)$")  # Dash-dot black line

plt.xlabel("h (Subdomain Element Size)")
plt.ylabel("Relative L2 Norm")
plt.title("Relative L2 Norm vs. h (Subdomain Solver)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "relative_L2_vs_h_subdomain.png"), dpi=300)
plt.show()

# --- Plot 2: Convergence of L2 and H1 Norms with O(h^2) and O(h) References ---
plt.figure(figsize=(8, 6))

for i, (Nx, Ny) in enumerate(global_mesh_sizes):
    subset = df[(df['global_mesh_x'] == Nx) & (df['global_mesh_y'] == Ny)]
    h_subdomain = 1 / subset['sub_mesh_x']
    plt.loglog(h_subdomain, subset['L2_norm'], marker='s', linestyle='-', color=colors[i % len(colors)],
               label=f"L2 Norm (Global {Nx}x{Ny})")
    plt.loglog(h_subdomain, subset['H1_norm'], marker='^', linestyle='--', color=colors[i % len(colors)],
               label=f"H1 Norm (Global {Nx}x{Ny})")

# Add reference lines for O(h^2) and O(h)
o_h2 = 10 * h_ref**2  # Quadratic convergence
o_h1 = 10 * h_ref  # Linear convergence
plt.loglog(h_ref, o_h2, 'k--', label=r"$O(h^2)$")  # Dashed black line
plt.loglog(h_ref, o_h1, 'k-.', label=r"$O(h)$")  # Dash-dot black line

plt.xlabel("h (Subdomain Element Size)")
plt.ylabel("Error Norms")
plt.title("Convergence Plot: L2 and H1 Norms vs. h (Subdomain Solver)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "convergence_subdomain.png"), dpi=300)
plt.show()
