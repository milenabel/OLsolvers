import json
import os
import matplotlib.pyplot as plt

# Define mesh sizes and base directories
mesh_sizes = [(20, 20), (24, 24), (48, 48)]
results_dir = "../results/deeponets/2"
figs_base = "../figs"

# Loop over mesh sizes
for mesh_size in mesh_sizes:
    Nx, Ny = mesh_size
    print(f"Plotting for mesh {Nx}x{Ny}...")

    # Create figure directory
    figs_dir = os.path.join(figs_base, f"deeponet_{Nx}x{Ny}")
    os.makedirs(figs_dir, exist_ok=True)

    # File paths
    results_filename = os.path.join(results_dir, f"deeponet_results_FEM_{Nx}x{Ny}.json")
    plot_filename = os.path.join(figs_dir, f"train_vs_test_l2_error_{Nx}x{Ny}.png")

    # Check if results file exists
    if not os.path.exists(results_filename):
        print(f"[Warning] File not found: {results_filename}")
        continue

    # Load results
    try:
        with open(results_filename, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"[Error] Could not load {results_filename}: {e}")
        continue

    training_results = results["training_results"]

    # Extract values
    epochs = [entry["epoch"] for entry in training_results]
    train_l2_errors = [entry["L2_error_train"] for entry in training_results]
    test_l2_errors = [entry["L2_error_test"] for entry in training_results]

    # Generalization errors (from final epoch)
    last_entry = training_results[-1]
    gen_error_fem = last_entry.get("generalization_error_fem", None)
    gen_error_true = last_entry.get("generalization_error_true", None)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_l2_errors, marker="o", linestyle="-", markersize=3, label="Train L2 Error", color="blue")
    plt.plot(epochs, test_l2_errors, marker="s", linestyle="--", markersize=3, label="Test L2 Error", color="red")

    if gen_error_fem is not None:
        plt.axhline(y=gen_error_fem, color="green", linestyle=":", label=f"Gen Error (FEM): {gen_error_fem:.4f}")
    if gen_error_true is not None:
        plt.axhline(y=gen_error_true, color="purple", linestyle="--", label=f"Gen Error (True): {gen_error_true:.4f}")

    plt.xlabel("Epochs")
    plt.ylabel("L2 Error")
    plt.title(f"DeepONet Train vs Test L2 Error ({Nx}x{Ny} mesh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    print(f"[Saved] Plot to {plot_filename}")
