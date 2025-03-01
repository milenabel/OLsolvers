import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.optimize import minimize
from sklearn.decomposition import PCA

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
T = 2  # Total time duration (ensures a few full periods)
high_res_t = np.linspace(0, T, 1000)  # High-resolution time points

# Generate ground-truth signal
ground_truth = fourier_series(high_res_t, a0, a, b, omega)

# Add Gaussian noise to the signal
def add_noise(signal, noise_level):
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

# Noise levels to test
noise_levels = [0.05, 0.1, 0.2]  # 5%, 10%, 20% noise

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

# Function to perform Robust PCA (RPCA)
def singular_value_thresholding(X, tau):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_thresholded = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresholded) @ VT

def soft_thresholding(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def robust_pca(V, lambda_val, max_iter=500, tol=1e-6):
    L, S = np.zeros_like(V), np.zeros_like(V)
    mu = np.prod(V.shape) / (4 * np.sum(np.abs(V)))
    
    for _ in range(max_iter):
        L_new = singular_value_thresholding(V - S, mu)
        S_new = soft_thresholding(V - L_new, lambda_val * mu)
        error = np.linalg.norm(V - (L_new + S_new), 'fro') / np.linalg.norm(V, 'fro')
        if error < tol:
            break
        L, S = L_new, S_new

    return L, S

# Number of measurements (sensors)
p = 100  # Fixed number of measurements for this part

# Test compressed sensing and RPCA for different noise levels
for noise_level in noise_levels:
    # Add noise to the ground truth signal
    noisy_signal = add_noise(ground_truth, noise_level)

    # Perform compressed sensing reconstruction
    u_reconstructed, t_low_res, u_low_res = compressed_sensing_reconstruction(p, high_res_t, noisy_signal, omega)

    # Perform Robust PCA (RPCA)
    X = noisy_signal.reshape(-1, 1)  # Reshape signal into a matrix for RPCA
    L, S = robust_pca(X, lambda_val=0.3)  # Apply RPCA with λ=0.3
    u_rpca = L.flatten()  # Extract the low-rank (clean) component

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(high_res_t, ground_truth, label="Ground Truth", linewidth=2, color='black')
    plt.plot(high_res_t, noisy_signal, label=f"Noisy Signal (ζ={noise_level})", linewidth=1, color='gray', alpha=0.6)
    plt.plot(high_res_t, u_reconstructed, label=f"Compressed Sensing (p={p})", linestyle='--', linewidth=2, color='blue')
    plt.plot(high_res_t, u_rpca, label="Robust PCA (RPCA)", linestyle='-.', linewidth=2, color='red')
    plt.xlabel("Time (t)")
    plt.ylabel("u(t)")
    plt.title(f"Signal Reconstruction with Noise (ζ={noise_level})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figs_dir, f"p2c_reconstruction_noise_{noise_level}.png"))
    plt.show()

    # Calculate reconstruction errors
    cs_error = np.linalg.norm(u_reconstructed - ground_truth) / np.linalg.norm(ground_truth)
    rpca_error = np.linalg.norm(u_rpca - ground_truth) / np.linalg.norm(ground_truth)
    print(f"Noise level ζ={noise_level}:")
    print(f"  Compressed Sensing error: {cs_error:.4f}")
    print(f"  Robust PCA error: {rpca_error:.4f}")