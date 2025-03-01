import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_1_path = "../data/cleanExo_1.mat"

# Figures directory
figs_dir = "../figs"
os.makedirs(figs_dir, exist_ok=True)

with h5py.File(file_1_path, 'r') as f:
    time_steps = f['time'][()].flatten()  # Extract time vector
    x_positions = f['nvar01'][()]  # Shape (20, 8)
    y_positions = f['nvar02'][()]  # Shape (20, 8)

# Compute time differences
dt = np.diff(time_steps)[:, None]  # Shape (19, 1)

# Compute velocity using finite differences
vx = np.diff(x_positions, axis=0) / dt  # Shape (19, 8)
vy = np.diff(y_positions, axis=0) / dt  # Shape (19, 8)

# Center data separately for vx and vy
Vx_mean = np.mean(vx, axis=1, keepdims=True)
Vy_mean = np.mean(vy, axis=1, keepdims=True)

Vx_centered = vx - Vx_mean
Vy_centered = vy - Vy_mean

# Perform SVD separately for vx and vy
Ux, Sx, VTx = np.linalg.svd(Vx_centered, full_matrices=False)
Uy, Sy, VTy = np.linalg.svd(Vy_centered, full_matrices=False)

# Compute cumulative energy
cumulative_energy_x = np.cumsum(Sx) / np.sum(Sx)
cumulative_energy_y = np.cumsum(Sy) / np.sum(Sy)

# Plot singular value decay for both vx and vy
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, len(Sx) + 1), Sx, 'bo-', markersize=5)
axes[0].set_xlabel("Mode Index")
axes[0].set_ylabel("Singular Value")
axes[0].set_title("Singular Value Decay (vx)")
axes[0].grid()

axes[1].plot(range(1, len(Sy) + 1), Sy, 'ro-', markersize=5)
axes[1].set_xlabel("Mode Index")
axes[1].set_ylabel("Singular Value")
axes[1].set_title("Singular Value Decay (vy)")
axes[1].grid()

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "p1_svd_vx_vy.png"))
plt.show()

# Plot cumulative energy for both vx and vy
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, len(Sx) + 1), cumulative_energy_x, 'bo-', markersize=5)
axes[0].set_xlabel("Mode Index")
axes[0].set_ylabel("Cumulative Energy")
axes[0].set_title("Cumulative Energy (vx)")
axes[0].grid()

axes[1].plot(range(1, len(Sy) + 1), cumulative_energy_y, 'ro-', markersize=5)
axes[1].set_xlabel("Mode Index")
axes[1].set_ylabel("Cumulative Energy")
axes[1].set_title("Cumulative Energy (vy)")
axes[1].grid()

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "p1_cumenergy_vx_vy.png"))
plt.show()

# Plot first three dominant PCA modes for vx and vy
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i in range(3):
    # Extract spatial modes from right singular vectors (VTx, VTy)
    mode_x = VTx[i, :].reshape(8, 1)  # Corrected
    mode_y = VTy[i, :].reshape(8, 1)  # Corrected

    # Compute magnitude
    magnitude_x = np.abs(mode_x)
    magnitude_y = np.abs(mode_y)

    # Plot vx modes
    scatter_x = axes[0, i].scatter(range(8), magnitude_x, c=magnitude_x, cmap="coolwarm", edgecolors='k')
    axes[0, i].set_title(f"vx Mode {i+1}")
    axes[0, i].set_xlabel("Spatial Point Index")
    axes[0, i].set_ylabel("Magnitude")
    plt.colorbar(scatter_x, ax=axes[0, i])

    # Plot vy modes
    scatter_y = axes[1, i].scatter(range(8), magnitude_y, c=magnitude_y, cmap="coolwarm", edgecolors='k')
    axes[1, i].set_title(f"vy Mode {i+1}")
    axes[1, i].set_xlabel("Spatial Point Index")
    axes[1, i].set_ylabel("Magnitude")
    plt.colorbar(scatter_y, ax=axes[1, i])

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, "p1_pca_modes_vx_vy.png"))
plt.show()

# Function to add noise separately to vx and vy
def add_noise_separate(Vx, Vy, noise_level, method="random"):
    """Adds noise separately to vx and vy."""
    Vx_noisy = Vx.copy()
    Vy_noisy = Vy.copy()
    
    rand_noise_x = np.random.randn(*Vx.shape)
    rand_noise_y = np.random.randn(*Vy.shape)

    if method == "random":
        mask_x = np.random.rand(*Vx.shape) < 0.8
        mask_y = np.random.rand(*Vy.shape) < 0.8
        Vx_noisy[mask_x] += noise_level * rand_noise_x[mask_x] * Vx_noisy[mask_x]
        Vy_noisy[mask_y] += noise_level * rand_noise_y[mask_y] * Vy_noisy[mask_y]

    elif method == "localized":
        Vx_noisy[:4, :] += noise_level * rand_noise_x[:4, :] * Vx_noisy[:4, :]
        Vy_noisy[:4, :] += noise_level * rand_noise_y[:4, :] * Vy_noisy[:4, :]
    
    return Vx_noisy, Vy_noisy

# Perform PCA on noisy data
def perform_pca_on_noisy_data(Vx_noisy, Vy_noisy, noise_type, noise_level):
    """Performs PCA on noisy vx and vy separately and compares with original."""
    Vx_noisy_centered = Vx_noisy - np.mean(Vx_noisy, axis=1, keepdims=True)
    Vy_noisy_centered = Vy_noisy - np.mean(Vy_noisy, axis=1, keepdims=True)

    Ux_noisy, Sx_noisy, VTx_noisy = np.linalg.svd(Vx_noisy_centered, full_matrices=False)
    Uy_noisy, Sy_noisy, VTy_noisy = np.linalg.svd(Vy_noisy_centered, full_matrices=False)

    # Compute cumulative energy for noisy data
    cumulative_energy_x_noisy = np.cumsum(Sx_noisy) / np.sum(Sx_noisy)
    cumulative_energy_y_noisy = np.cumsum(Sy_noisy) / np.sum(Sy_noisy)

    # Plot singular values and cumulative energy together
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # vx Singular Value Decay & Cumulative Energy
    ax1 = axes[0]
    ax2 = ax1.twinx()  # Create second y-axis

    ax1.plot(range(1, len(Sx) + 1), Sx, 'bo-', markersize=5, label="Original Singular Values")
    ax1.plot(range(1, len(Sx_noisy) + 1), Sx_noisy, 'ro-', markersize=5, label=f"Noisy Singular Values ({noise_type}, ζ={noise_level})")
    ax2.plot(range(1, len(cumulative_energy_x_noisy) + 1), cumulative_energy_x_noisy, 'g^-', markersize=5, label="Cumulative Energy")

    ax1.set_xlabel("Mode Index")
    ax1.set_ylabel("Singular Value", color='b')
    ax2.set_ylabel("Cumulative Energy", color='g')
    ax1.set_title(f"Singular Value Decay & Cumulative Energy (vx, {noise_type} Noise, ζ={noise_level})")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid()

    # vy Singular Value Decay & Cumulative Energy
    ax3 = axes[1]
    ax4 = ax3.twinx()  # Create second y-axis

    ax3.plot(range(1, len(Sy) + 1), Sy, 'bo-', markersize=5, label="Original Singular Values")
    ax3.plot(range(1, len(Sy_noisy) + 1), Sy_noisy, 'ro-', markersize=5, label=f"Noisy Singular Values ({noise_type}, ζ={noise_level})")
    ax4.plot(range(1, len(cumulative_energy_y_noisy) + 1), cumulative_energy_y_noisy, 'g^-', markersize=5, label="Cumulative Energy")

    ax3.set_xlabel("Mode Index")
    ax3.set_ylabel("Singular Value", color='b')
    ax4.set_ylabel("Cumulative Energy", color='g')
    ax3.set_title(f"Singular Value Decay & Cumulative Energy (vy, {noise_type} Noise, ζ={noise_level})")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper right")
    ax3.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f"p1_{noise_type}_noise_z{int(noise_level*100)}_vx_vy.png"))
    plt.show()

# Noise levels
noise_levels = [0.05, 0.1, 0.2]

# Apply noise and perform PCA
for noise_type in ["random", "localized"]:
    for noise_level in noise_levels:
        Vx_noisy, Vy_noisy = add_noise_separate(vx, vy, noise_level, method=noise_type)
        perform_pca_on_noisy_data(Vx_noisy, Vy_noisy, noise_type, noise_level)

# Function to perform Robust PCA
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


print(f"Sum of Vy_noisy: {np.sum(Vy_noisy)}, Min: {np.min(Vy_noisy)}, Max: {np.max(Vy_noisy)}")


# Define noise levels and λ values to test
noise_levels = [0.05, 0.1, 0.2]
lambda_vals = [0.3]  

# Apply noise and run RPCA for different noise levels
for noise_type in ["random", "localized"]:
    for noise_level in noise_levels:
        Vx_noisy, Vy_noisy = add_noise_separate(vx, vy, noise_level, method=noise_type)
        Lx, Sx_rpca = robust_pca(Vx_noisy, lambda_vals[0])
        Ly, Sy_rpca = robust_pca(Vy_noisy, lambda_vals[0])

        # Compute SVD on cleaned (low-rank) data
        Ux_clean, Sx_clean, _ = np.linalg.svd(Lx, full_matrices=False)
        Uy_clean, Sy_clean, _ = np.linalg.svd(Ly, full_matrices=False)

        # Compute cumulative energy after RPCA
        cumulative_energy_x_clean = np.cumsum(Sx_clean) / np.sum(Sx_clean)
        print(f"Sum of Sy_clean: {np.sum(Sy_clean)}, Min: {np.min(Sy_clean)}, Max: {np.max(Sy_clean)}")
        cumulative_energy_y_clean = np.cumsum(Sy_clean) / np.sum(Sy_clean)

        # Generate plots for each noise level
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # vx Singular Value Decay
        ax1 = axes[0]
        ax2 = ax1.twinx()

        ax1.plot(range(1, len(Sx) + 1), Sx, 'bo-', markersize=5, label="Original Singular Values")
        ax1.plot(range(1, len(Sx_clean) + 1), Sx_clean, 'ro-', markersize=5, label="After RPCA (Cleaned)")
        ax2.plot(range(1, len(cumulative_energy_x_clean) + 1), cumulative_energy_x_clean, 'g^-', markersize=5, label="Cumulative Energy")

        ax1.set_xlabel("Mode Index")
        ax1.set_ylabel("Singular Value", color='r')
        ax2.set_ylabel("Cumulative Energy", color='g')
        ax1.set_title(f"Singular Value Decay (vx, {noise_type} Noise, ζ={noise_level}, λ={lambda_vals[0]})")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1, labels1, loc="upper left")
        ax2.legend(handles2, labels2, loc="upper right")

        ax1.grid()

        # vy Singular Value Decay
        ax3 = axes[1]
        ax4 = ax3.twinx()

        ax3.plot(range(1, len(Sy) + 1), Sy, 'bo-', markersize=5, label="Original Singular Values")
        ax3.plot(range(1, len(Sy_clean) + 1), Sy_clean, 'ro-', markersize=5, label="After RPCA (Cleaned)")
        ax4.plot(range(1, len(cumulative_energy_y_clean) + 1), cumulative_energy_y_clean, 'g^-', markersize=5, label="Cumulative Energy")

        ax3.set_xlabel("Mode Index")
        ax3.set_ylabel("Singular Value", color='r')
        ax4.set_ylabel("Cumulative Energy", color='g')
        ax3.set_title(f"Singular Value Decay (vy, {noise_type} Noise, ζ={noise_level}, λ={lambda_vals[0]})")

        handles3, labels3 = ax3.get_legend_handles_labels()
        handles4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(handles3, labels3, loc="upper left")
        ax4.legend(handles4, labels4, loc="upper right")

        ax3.grid()

        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"p1_rpca_{noise_type}_z{int(noise_level*100)}_lambda{lambda_vals[0]}.png"))
        plt.show()
