import json
import os
import matplotlib.pyplot as plt

figs_dir = "../figs/deeponet_1"
os.makedirs(figs_dir, exist_ok=True)

# Load the results
with open("../results/deeponets/1/deeponet_results.json", "r") as f:
    results = json.load(f)

# Extract training results
training_results = results["training_results"]
epochs = [entry["epoch"] for entry in training_results]
train_l2_errors = [entry["L2_error_train"] for entry in training_results]
test_l2_errors = [entry["L2_error_test"] for entry in training_results]

# Extract generalization error
generalization_error = results["generalization_error"]

# Plot Train vs Test L2 Errors Over Epochs
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_l2_errors, marker="o", linestyle="-", markersize=3, label="Train L2 Error", color="blue")
plt.plot(epochs, test_l2_errors, marker="s", linestyle="--", markersize=3, label="Test L2 Error", color="red")
plt.axhline(y=generalization_error, color="green", linestyle="dashed", label=f"Generalization Error: {generalization_error:.6f}")
plt.xlabel("Epochs")
plt.ylabel("L2 Error")
plt.title("DeepONet Train vs Test L2 Error Over Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(figs_dir, "train_vs_test_l2_error_1000fs.png")) 
plt.show()

# Extract final predictions
final_predictions = results["final_predictions"]
u_pred_train = final_predictions["u_pred_train"]
u_pred_test = final_predictions["u_pred_test"]
u_true_train = final_predictions["u_true_train"]
u_true_test = final_predictions["u_true_test"]

# Print summary of final predictions
print(f"Generalization Error: {generalization_error:.6f}")
print("Final Predictions Summary:")
print(f"  u_pred_train (first 5 values): {u_pred_train[0][:5]}")
print(f"  u_pred_test (first 5 values): {u_pred_test[0][:5]}")
print(f"  u_true_train (first 5 values): {u_true_train[0][:5]}")
print(f"  u_true_test (first 5 values): {u_true_test[0][:5]}")
