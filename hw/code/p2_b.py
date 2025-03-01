import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.optimize import minimize

# Figures directory
figs_dir = "../figs"
os.makedirs(figs_dir, exist_ok=True)

# Define Fourier series signal
def fourier_series(t, a0, a, b, omega):
    N = len(a)
    result = a0 + sum(a[i] * np.cos((i+1) * omega * t) + b[i] * np.sin((i+1) * omega * t) for i in range(N))
    return result

# Parameters
N = 3  # Number of Fourier modes
a0 = 1
a = np.array([2, -1.5, 0.5])  # Arbitrary coefficients
b = np.array([-1, 1.2, -0.8])
omega = 2 * np.pi  # Base frequency
T = 1  # Total time duration set to one period
high_res_t = np.linspace(0, T, 1000)  # High-resolution time points

# Generate ground-truth signal
ground_truth = fourier_series(high_res_t, a0, a, b, omega)

# Plot ground-truth signal
plt.figure(figsize=(10, 4))
plt.plot(high_res_t, ground_truth, label="Ground Truth", linewidth=2)
plt.xlabel("Time (t)")
plt.ylabel("u(t)")
plt.title("Fourier Series Signal (One Period)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(figs_dir, "p2b_ground_truth.png"))
plt.show()

# Function to perform compressed sensing reconstruction
def compressed_sensing_reconstruction(p, t_high_res, ground_truth, omega):
    # Randomly select p time points for measurements
    p_indices = np.random.choice(len(t_high_res), p, replace=False)
    t_low_res = t_high_res[p_indices]
    u_low_res = ground_truth[p_indices]

    # Define the optimization problem for compressed sensing
    def objective(x):
        # x is the sparse representation in the DCT domain
        # Reconstruct the signal from the sparse representation
        u_reconstructed = idct(x, norm='ortho')
        # Calculate the error between the reconstructed signal and the low-res measurements
        error = np.linalg.norm(u_reconstructed[p_indices] - u_low_res)
        return error

    # Initial guess for the sparse representation (zeros)
    x0 = np.zeros(len(t_high_res))

    # Perform the optimization to find the sparse representation
    result = minimize(objective, x0, method='L-BFGS-B', tol=1e-8)
    x_opt = result.x

    # Reconstruct the high-resolution signal from the optimal sparse representation
    u_reconstructed = idct(x_opt, norm='ortho')

    return u_reconstructed, t_low_res, u_low_res

# Number of measurements (sensors)
p_values = [10, 20, 100, 500, 1000]  # Try different values of p

# Reconstruct the signal for different p values
for p in p_values:
    u_reconstructed, t_low_res, u_low_res = compressed_sensing_reconstruction(p, high_res_t, ground_truth, omega)

    # Plot the results
    plt.figure(figsize=(10, 4))
    plt.plot(high_res_t, ground_truth, label="Ground Truth", linewidth=2)
    plt.plot(high_res_t, u_reconstructed, label=f"Reconstructed (p={p})", linestyle='--', linewidth=2)
    plt.scatter(t_low_res, u_low_res, color='red', label=f"Measurements (p={p})", zorder=5)
    plt.xlabel("Time (t)")
    plt.ylabel("u(t)")
    plt.title(f"Compressed Sensing Reconstruction (p={p})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figs_dir, f"p2b_reconstruction_p_{p}.png"))
    plt.show()

    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(u_reconstructed - ground_truth) / np.linalg.norm(ground_truth)
    print(f"Reconstruction error for p={p}: {reconstruction_error:.4f}")

# Create a single plot with all reconstructions and ground truth
plt.figure(figsize=(12, 8))
plt.plot(high_res_t, ground_truth, label="Ground Truth", linewidth=3, color='black', linestyle='-', alpha=0.8)

# Extend the colors and linestyles lists to match the number of p_values
colors = ['red', 'blue', 'green', 'purple', 'orange']  # Added 'orange' for the 5th p_value
linestyles = ['--', '-.', ':', '-', '--']  # Added '--' for the 5th p_value

for i, p in enumerate(p_values):
    u_reconstructed, t_low_res, u_low_res = compressed_sensing_reconstruction(p, high_res_t, ground_truth, omega)
    
    # Plot the reconstructed signal
    plt.plot(high_res_t, u_reconstructed, label=f"Reconstructed (p={p})", 
             linestyle=linestyles[i], linewidth=2, color=colors[i], alpha=0.8)

    # Plot the low-resolution measurements (scatter points)
    plt.scatter(t_low_res, u_low_res, color=colors[i], marker='o', s=50, edgecolor='black', zorder=5)

# Add labels, title, and legend
plt.xlabel("Time (t)", fontsize=14)
plt.ylabel("u(t)", fontsize=14)
plt.title("Compressed Sensing Reconstructions for Different p Values (One Period)", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the high-resolution plot
plt.savefig(os.path.join(figs_dir, "p2b_combined_reconstructions.png"), dpi=300, bbox_inches='tight')

# Show the plot
plt.show()