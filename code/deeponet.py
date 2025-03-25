import jax
import numpy as np
import jax.numpy as jnp
import optax
import flax.linen as nn
import json
import os
import gc  # Garbage collection
import psutil
from jax import random
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

# Choose between "manufactured" and "FEM"
solution_type = "FEM"  

# Function to check memory usage
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

# Load dataset based on solution type
if solution_type == "manufactured":
    dataset_path = f"../results/deeponets/1/dataset1000.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
elif solution_type == "FEM":
    mesh_size = (20, 20)  # You can loop over or change this manually
    fem_files = f"../results/general/poisson_dataset_{mesh_size[0]}x{mesh_size[1]}.json"
    if not os.path.exists(fem_files):
        raise FileNotFoundError(f"FEM dataset not found: {fem_files}")
    
    with open(fem_files, "r") as f:
        data = json.load(f)

    print(f"[FEM Loader] Loaded {len(data['u_values'])} samples with {mesh_size[0]}x{mesh_size[1]} mesh.")

# Extract data
if solution_type == "FEM":
    u_values = jnp.array(data["u_values"], dtype=jnp.float32)
    f_values = jnp.array(data["f_values"], dtype=jnp.float32)
    num_samples, grid_size = u_values.shape
    Lx, Ly = data["domain_size"]
    a_samples = f_values

    # Create trunk input grid (same for all samples)
    x_grid = jnp.linspace(0, Lx, int(np.sqrt(grid_size)), dtype=jnp.float32)
    y_grid = jnp.linspace(0, Ly, int(np.sqrt(grid_size)), dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    x_inputs = jnp.hstack([X.flatten()[:, None], Y.flatten()[:, None]])  # (grid_size, 2)
    x_inputs = jnp.tile(x_inputs[None, :, :], (num_samples, 1, 1))        # (num_samples, grid_size, 2)
    
    with open("../results/deeponets/1/dataset1000.json", "r") as f:
        manufactured_data = json.load(f)

    u_exact_all = jnp.array([sample["u_values"] for sample in manufactured_data], dtype=jnp.float32)

    # Make sure to split the u_exact like you did for the FEM u_values
    _, _, u_exact_train, u_exact_heldout = train_test_split(
        a_samples, u_exact_all, test_size=0.2, random_state=42
    )
    _, u_exact_heldout = train_test_split(
        u_exact_train, test_size=0.125, random_state=42
    )

else:
    x_inputs = jnp.tile(x_inputs[None, :, :], (num_samples, 1, 1))
    u_values = jnp.array([sample["u_values"] for sample in dataset], dtype=jnp.float32)
    num_samples = len(dataset)
    a_samples = jnp.array([sample["a_k"][:50] for sample in dataset], dtype=jnp.float32)  # 50 basis functions
    
    # Generate input grid only for manufactured case
    Lx, Ly = 12.0, 12.0  
    x_grid = jnp.linspace(0, Lx, int(jnp.sqrt(grid_size)), dtype=jnp.float32)  
    y_grid = jnp.linspace(0, Ly, int(jnp.sqrt(grid_size)), dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    X = X.flatten()[:, None]  
    Y = Y.flatten()[:, None]
    x_inputs = jnp.hstack([X, Y])

# Print dataset info
print(f"Dataset loaded: {num_samples} samples from {solution_type} solutions")
if a_samples is not None:
    print(f"a_samples shape: {a_samples.shape}")
print(f"u_values shape: {u_values.shape}")
print_memory_usage("After Loading Dataset")
print(f"x_inputs shape: {x_inputs.shape}, u_values shape: {u_values.shape}")
print_memory_usage("After Processing Inputs")    


# Split data into training (80%) and testing sets (20%)
if a_samples is not None:
    a_train, a_test, u_train, u_test, x_train, x_test = train_test_split(
        a_samples, u_values, x_inputs, test_size=0.2, random_state=42
    )
else:
    u_train, u_test, x_train, x_test = train_test_split(
        u_values, x_inputs, test_size=0.2, random_state=42
    )
    a_train = a_test = None

# Define held-out dataset (10% of total samples) for generalization error
if a_samples is not None:
    a_train, a_heldout, u_train, u_heldout, x_train, x_heldout = train_test_split(
        a_train, u_train, x_train, test_size=0.125, random_state=42
    )
else:
    u_train, u_heldout, x_train, x_heldout = train_test_split(
        u_train, x_train, test_size=0.125, random_state=42
    )
    a_heldout = None

print(f"Training samples: {u_train.shape[0]}, Testing samples: {u_test.shape[0]}, Held-out samples: {u_heldout.shape[0]}")
print_memory_usage("After Train-Test Split")

# Define DeepONet model with tanh activation
class DeepONet(nn.Module):
    trunk_layers: int
    branch_layers: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.trunk_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[nn.Dense(self.hidden_dim), nn.tanh] * (self.trunk_layers - 1),
            nn.Dense(self.output_dim)
        ])

        self.branch_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[nn.Dense(self.hidden_dim), nn.tanh] * (self.branch_layers - 1),
            nn.Dense(self.output_dim)
        ])
    
    def __call__(self, x, a=None):
        trunk_out = jax.vmap(self.trunk_net)(x)
        if a is not None:
            branch_out = self.branch_net(a)[:, None, :]
            return jnp.sum(trunk_out * branch_out, axis=-1)
        else:
            return trunk_out.sum(axis=-1)

# Initialize model
model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
key = random.PRNGKey(0)

print_memory_usage("Before Model Initialization")
params = model.init(key, x_train, a_train if a_train is not None else x_train)
print_memory_usage("After Model Initialization")

# Loss function
def loss_fn(params, x, a, true_u):
    pred_u = model.apply(params, x, a)
    return jnp.mean((pred_u - true_u) ** 2)

# Define optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, x, a, true_u):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, a, true_u)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Tracking results efficiently
training_results = []
final_predictions = {}

# Training loop
num_epochs = 100000  # Train for 100k epochs
batch_size = 16  
if solution_type == "manufactured":
    num_batches = len(x_train) // batch_size
else:
    num_batches = max(1, len(x_train) // batch_size)

print_memory_usage("Before Training Loop")

for epoch in range(num_epochs):
    loss = 0.0  # Initialize loss for this epoch
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        x_batch = x_train[start_idx:end_idx]
        a_batch = a_train[start_idx:end_idx] if a_train is not None else None
        u_batch = u_train[start_idx:end_idx]

        params, opt_state, loss = train_step(params, opt_state, x_batch, a_batch, u_batch)

    # Compute predictions
    u_pred_train = model.apply(params, x_train, a_train)
    u_pred_test = model.apply(params, x_test, a_test)

    # Compute L2 errors
    L2_error_train = jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)
    L2_error_test = jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)

    # Compute generalization error on held-out data
    u_pred_heldout = model.apply(params, x_heldout, a_heldout)
    L2_error_heldout = jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout)
    if solution_type == "manufactured":
        generalization_error = float(L2_error_heldout)

        # Store results
        training_results.append({
            "epoch": epoch,
            "L2_error_train": float(L2_error_train),
            "L2_error_test": float(L2_error_test),
            "generalization_error": generalization_error
        })
    elif solution_type == "FEM":
        # Compute true generalization error vs manufactured solution
        generalization_error_fem = float(jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout))
        generalization_error_true = float(jnp.linalg.norm(u_pred_heldout - u_exact_heldout) / jnp.linalg.norm(u_exact_heldout))

        training_results.append({
            "epoch": epoch,
            "L2_error_train": float(L2_error_train),
            "L2_error_test": float(L2_error_test),
            "generalization_error_fem": generalization_error_fem,
            "generalization_error_true": generalization_error_true
        })

    if epoch % 100 == 0:
        print_memory_usage(f"Epoch {epoch}")
        print(f"Epoch {epoch}, Loss: {loss:.6f}, L2 Error Train: {L2_error_train:.6f}, L2 Error Test: {L2_error_test:.6f}")

# Save results
if solution_type == "manufactured":
    results_filename = f"../results/deeponets/2/deeponet_results_{solution_type}.json"
elif solution_type == "FEM":
    results_filename = f"../results/deeponets/2/deeponet_results_{solution_type}.json"

results_data = {
    "training_results": training_results,
    "solution_type": solution_type
}

with open(results_filename, "w") as f:
    json.dump(results_data, f)


print(f"Training complete using {solution_type} solutions. Results saved to {results_filename}.")
print_memory_usage("After Training")
