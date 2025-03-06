import json
import os
import matplotlib.pyplot as plt

figs_dir = "../figs/deeponet_1"
os.makedirs(figs_dir, exist_ok=True)

# Load the results
with open("../results/deeponets/1/deeponet_results.json", "r") as f:
    training_results = json.load(f)

# Extract epochs and correct L2 error keys
epochs = [entry["epoch"] for entry in training_results]
train_l2_errors = [entry["L2_error_train"] for entry in training_results]  # Corrected key
test_l2_errors = [entry["L2_error_test"] for entry in training_results]  # Corrected key

# Plot Train vs Test L2 Errors Over Epochs
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_l2_errors, marker="o", linestyle="-", markersize=3, label="Train L2 Error", color="blue")
plt.plot(epochs, test_l2_errors, marker="s", linestyle="--", markersize=3, label="Test L2 Error", color="red")
plt.xlabel("Epochs")
plt.ylabel("L2 Error")
plt.title("DeepONet Train vs Test L2 Error Over Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(figs_dir, "train_vs_test_l2_error_1000fs.png")) 
plt.show()
