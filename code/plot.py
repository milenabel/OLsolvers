import numpy as np
import matplotlib.pyplot as plt
import os

figs_dir = "../figs/general"
os.makedirs(figs_dir, exist_ok=True)

# Given mesh sizes and corresponding error norms
mesh_sizes = np.array([3, 4, 6, 8, 12, 24, 48, 96], dtype=float)
h_values = 1 / mesh_sizes  # Compute h values (element size)

L2_norms = np.array([0.4713439166214983, 0.28363365893389303, 0.13211792838711256, 
                     0.07552991957736872, 0.03395708500318974, 0.008547804140086358, 
                     0.0021406180988177525, 0.0005353838530320575])

H1_norms = np.array([0.18255071394112415, 0.10772812494689904, 0.04947588749611865, 
                     0.028144230051207936, 0.012608213332163016, 0.003167001036735138, 
                     0.000792684435027786, 0.00019822949389266875])

Relative_L2_norms = np.array([0.0942687833242997, 0.05238686203755397, 0.02304898069163268, 
                              0.012916045058890805, 0.0057245337530366395, 0.001428708289171142, 
                              0.0003570244887483297, 8.924657008244771e-05])

# Plot Relative L2 Norm vs. h
plt.figure(figsize=(8, 6))
plt.loglog(h_values, Relative_L2_norms, marker='o', linestyle='-', label="Relative L2 Norm")

# Add a reference dashed line for O(h^2) slope
h_ref = np.array([h_values[0], h_values[-1]])  # Use first and last h values
slope_ref = Relative_L2_norms[0] * (h_ref / h_values[0])**2  # Scale to match initial value

plt.loglog(h_ref, slope_ref, 'k--', label=r"$O(h^2)$ (Reference Slope)")

plt.xlabel("h (Element Size)")
plt.ylabel("Relative L2 Norm")
plt.title("Relative L2 Norm vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "relative_L2_vs_h.png"), dpi=300) 
plt.show()

# Plot Convergence of L2 and H1 norms
plt.figure(figsize=(8, 6))
plt.loglog(h_values, L2_norms, marker='s', linestyle='-', label="L2 Norm")
plt.loglog(h_values, H1_norms, marker='^', linestyle='-', label="H1 Norm")

# Add reference dashed lines for O(h^2) and O(h) slopes
slope_L2_ref = L2_norms[0] * (h_ref / h_values[0])**2  # Second-order slope for L2
slope_H1_ref = H1_norms[0] * (h_ref / h_values[0])**1  # First-order slope for H1

plt.loglog(h_ref, slope_L2_ref, 'k--', label=r"$O(h^2)$ (Reference Slope for L2)")
plt.loglog(h_ref, slope_H1_ref, 'r--', label=r"$O(h)$ (Reference Slope for H1)")

plt.xlabel("h (Element Size)")
plt.ylabel("Error Norms")
plt.title("Convergence Plot: L2 and H1 Norms vs. h")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig(os.path.join(figs_dir, "convergence.png"), dpi=300) 
plt.show()