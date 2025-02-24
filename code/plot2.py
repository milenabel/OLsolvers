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

# Compute min/max h values for reference lines
h_min = 1 / df['sub_mesh_x'].max()
h_max = 1 / df['sub_mesh_x'].min()
h_ref = np.array([h_max, h_min])

# Select a reasonable reference error from the middle of the data
ref_index = len(df) // 2
ref_h = 1 / df.iloc[ref_index]['sub_mesh_x']
ref_L2 = df.iloc[ref_index]['Relative_L2_norm']
ref_H1 = df.iloc[ref_index]['H1_norm']

# Define reference lines for O(h^2) and O(h)
o_h2 = ref_L2 * (h_ref / ref_h)**2  # Quadratic convergence
o_h1 = ref_H1 * (h_ref / ref_h)  # Linear convergence

# --- Plot 1: Relative L2 Norm vs. h with O(h^2) Reference ---
plt.figure(figsize=(8, 6))

for i, (Nx, Ny) in enumerate(global_mesh_sizes):
    subset = df[(df['global_mesh_x'] == Nx) & (df['global_mesh_y'] == Ny)]
    h_subdomain = 1 / subset['sub_mesh_x']
    plt.loglog(h_subdomain, subset['Relative_L2_norm'], marker='o', linestyle='-', color=colors[i % len(colors)],
               label=f"Global Mesh: {Nx}x{Ny}")

# Plot reference lines
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

# Plot reference lines
plt.loglog(h_ref, o_h2, 'k--', label=r"$O(h^2)$")  # Dashed black line
plt.loglog(h_ref, o_h1, 'k-.', label=r"$O(h)$")  # Dash-dot black line

plt.xlabel("h (Subdomain Element Size)")
plt.ylabel("Error Norms")
plt.title("Convergence Plot: L2 and H1 Norms vs. h (Subdomain Solver)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "convergence_subdomain.png"), dpi=300)
plt.show()