import json
import os
import matplotlib.pyplot as plt

# Set the mesh size and paths
mesh_size = (20, 20)
results_dir = "../results/deeponets/2"
figs_dir = f"../figs/deeponet_{mesh_size[0]}x{mesh_size[1]}"
os.makedirs(figs_dir, exist_ok=True)

# File paths
results_filename = os.path.join(results_dir, "deeponet_results_FEM.json")
plot_filename = os.path.join(figs_dir, f"train_vs_test_l2_error_{mesh_size[0]}x{mesh_size[1]}.png")

# Load results
with open(results_filename, "r") as f:
    results = json.load(f)

training_results = results["training_results"]

# Extract training info
epochs = [entry["epoch"] for entry in training_results]
train_l2_errors = [entry["L2_error_train"] for entry in training_results]
test_l2_errors = [entry["L2_error_test"] for entry in training_results]

# Optional: Extract generalization errors from the last epoch (or average them)
last_entry = training_results[-1]
gen_error_fem = last_entry.get("generalization_error_fem", None)
gen_error_true = last_entry.get("generalization_error_true", None)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_l2_errors, marker="o", linestyle="-", markersize=3, label="Train L2 Error", color="blue")
plt.plot(epochs, test_l2_errors, marker="s", linestyle="--", markersize=3, label="Test L2 Error", color="red")

# Horizontal lines for generalization errors
if gen_error_fem is not None:
    plt.axhline(y=gen_error_fem, color="green", linestyle=":", label=f"Gen Error (FEM): {gen_error_fem:.4f}")
if gen_error_true is not None:
    plt.axhline(y=gen_error_true, color="purple", linestyle="--", label=f"Gen Error (True): {gen_error_true:.4f}")

plt.xlabel("Epochs")
plt.ylabel("L2 Error")
plt.title(f"DeepONet Train vs Test L2 Error ({mesh_size[0]}x{mesh_size[1]} mesh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_filename)
plt.show()

print(f"[Saved] Plot to {plot_filename}")
