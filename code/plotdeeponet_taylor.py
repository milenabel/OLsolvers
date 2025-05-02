import json
import os
import matplotlib.pyplot as plt
import itertools

# === SETTINGS ===
Nx, Ny = 20, 20
lambda_values = [0.1, 1.0, 10.0]
results_dir = "../results/deeponets/taylor"
figs_dir = f"../figs/taylor/deeponet_taylor_{Nx}x{Ny}"
os.makedirs(figs_dir, exist_ok=True)
plot_filename = os.path.join(figs_dir, f"l2_errors_vs_epochs_{Nx}x{Ny}.png")

# Color cycle
color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)

# === Plot Initialization ===
plt.figure(figsize=(10, 6))

for lam in lambda_values:
    results_filename = os.path.join(results_dir, f"deeponet_results_FEM_lambda{lam}_{Nx}x{Ny}.json")

    if not os.path.exists(results_filename):
        print(f"[Warning] File not found: {results_filename}")
        continue

    with open(results_filename, "r") as f:
        results = json.load(f)

    training_results = results["training_results"]
    epochs = [entry["epoch"] for entry in training_results]
    train_l2_errors = [entry["L2_error_train"] for entry in training_results]
    test_l2_errors = [entry["L2_error_test"] for entry in training_results]

    last_entry = training_results[-1]
    gen_error_fem = last_entry.get("generalization_error_fem", None)
    gen_error_true = last_entry.get("generalization_error_true", None)

    # Plot with unique colors
    color_train = next(color_cycle)
    color_test = next(color_cycle)
    color_gen_fem = next(color_cycle)
    color_gen_true = next(color_cycle)

    plt.plot(epochs, train_l2_errors, linestyle="-", marker="o", markersize=2,
             label=f"Train λ={lam}", color=color_train)
    plt.plot(epochs, test_l2_errors, linestyle="--", marker="s", markersize=2,
             label=f"Test λ={lam}", color=color_test)

    if gen_error_fem is not None:
        plt.axhline(y=gen_error_fem, linestyle=":", color=color_gen_fem,
                    label=f"Gen Error (FEM) λ={lam}: {gen_error_fem:.4f}")
    if gen_error_true is not None:
        plt.axhline(y=gen_error_true, linestyle="-.", color=color_gen_true,
                    label=f"Gen Error (True) λ={lam}: {gen_error_true:.4f}")

# === Final Touches ===
plt.xlabel("Epochs")
plt.ylabel("L2 Error")
plt.title(f"Taylor-Enhanced DeepONet: All Errors ({Nx}x{Ny} mesh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_filename)
plt.close()

print(f"[Saved] Plot to {plot_filename}")
