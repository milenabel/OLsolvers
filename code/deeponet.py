import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import json
import os
import gc  # Garbage collection
import psutil
from jax import random
from sklearn.model_selection import train_test_split

# Function to check memory usage
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

# Load dataset
with open("../results/deeponets/1/dataset1000.json", "r") as f:
    dataset = json.load(f)

# Extract data
a_samples = jnp.array([sample["a_k"] for sample in dataset], dtype=jnp.float32)  
u_values = jnp.array([sample["u_values"] for sample in dataset], dtype=jnp.float32)  

# Print initial dataset sizes
print(f"Dataset loaded: {len(dataset)} samples")
print(f"a_samples shape: {a_samples.shape}, u_values shape: {u_values.shape}")
print_memory_usage("After Loading Dataset")

# Define domain size
num_samples, grid_size = u_values.shape
Lx, Ly = 12.0, 12.0  

# Generate input grid
x_grid = jnp.linspace(0, Lx, int(jnp.sqrt(grid_size)), dtype=jnp.float32)  
y_grid = jnp.linspace(0, Ly, int(jnp.sqrt(grid_size)), dtype=jnp.float32)
X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
X = X.flatten()[:, None]  
Y = Y.flatten()[:, None]
x_inputs = jnp.hstack([X, Y])

# Tile x_inputs
x_inputs = jnp.tile(x_inputs[None, :, :], (num_samples, 1, 1))

print(f"x_inputs shape: {x_inputs.shape}")
print_memory_usage("After Processing Inputs")

# Split data into training (80%) and testing sets (20%)
a_train, a_test, u_train, u_test, x_train, x_test = train_test_split(
    a_samples, u_values, x_inputs, test_size=0.2, random_state=42
)

# Define held-out dataset (10% of total samples) for generalization error
a_train, a_heldout, u_train, u_heldout, x_train, x_heldout = train_test_split(
    a_train, u_train, x_train, test_size=0.125, random_state=42
)

print(f"Training samples: {a_train.shape[0]}, Testing samples: {a_test.shape[0]}, Held-out samples: {a_heldout.shape[0]}")
print_memory_usage("After Train-Test Split")

# Define DeepONet model
class DeepONet(nn.Module):
    trunk_layers: int
    branch_layers: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.trunk_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            *[nn.Dense(self.hidden_dim), nn.relu] * (self.trunk_layers - 1),
            nn.Dense(self.output_dim)
        ])

        self.branch_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            *[nn.Dense(self.hidden_dim), nn.relu] * (self.branch_layers - 1),
            nn.Dense(self.output_dim)
        ])
    
    def __call__(self, x, a):
        trunk_out = jax.vmap(self.trunk_net)(x)
        branch_out = self.branch_net(a)[:, None, :]
        return jnp.sum(trunk_out * branch_out, axis=-1)

# Initialize model
model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
key = random.PRNGKey(0)

print_memory_usage("Before Model Initialization")
params = model.init(key, x_train, a_train)
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

# **Tracking results efficiently**
training_results = []
final_predictions = {}

# Training loop
num_epochs = 100000  # Train for 100k epochs
batch_size = 16  
num_batches = len(x_train) // batch_size

print_memory_usage("Before Training Loop")

for epoch in range(num_epochs):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        x_batch = x_train[start_idx:end_idx]
        a_batch = a_train[start_idx:end_idx]
        u_batch = u_train[start_idx:end_idx]

        params, opt_state, loss = train_step(params, opt_state, x_batch, a_batch, u_batch)

    # Compute predictions
    u_pred_train = model.apply(params, x_train, a_train)
    u_pred_test = model.apply(params, x_test, a_test)

    # Compute L2 errors
    L2_error_train = jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)
    L2_error_test = jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)

    # Store only necessary training results
    training_results.append({
        "epoch": epoch,
        "L2_error_train": float(L2_error_train),
        "L2_error_test": float(L2_error_test)
    })

    # Save final predictions at the last epoch
    if epoch == num_epochs - 1:
        final_predictions = {
            "u_pred_train": u_pred_train.tolist(),
            "u_pred_test": u_pred_test.tolist(),
            "u_true_train": u_train.tolist(),
            "u_true_test": u_test.tolist()
        }

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print_memory_usage(f"Epoch {epoch}")
        print(f"Epoch {epoch}, Loss: {loss:.6f}, L2 Error Train: {L2_error_train:.6f}, L2 Error Test: {L2_error_test:.6f}")
        print(f"  U_true_train (first 5): {u_train.flatten()[:5]}")
        print(f"  U_pred_train (first 5): {u_pred_train.flatten()[:5]}")
        print(f"  U_true_test (first 5): {u_test.flatten()[:5]}")
        print(f"  U_pred_test (first 5): {u_pred_test.flatten()[:5]}")

    # Delete variables only after printing to avoid NameError
    del u_pred_train, u_pred_test  
    gc.collect()  # Force garbage collection

# Compute generalization error on the held-out dataset **once**
u_pred_heldout = model.apply(params, x_heldout, a_heldout)
L2_error_heldout = jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout)
generalization_error = float(L2_error_heldout)

# Save results
results = {
    "training_results": training_results,
    "final_predictions": final_predictions,
    "generalization_error": generalization_error
}

with open("../results/deeponets/1/deeponet_results.json", "w") as f:
    json.dump(results, f)

print("Training complete. Results saved to deeponet_results.json")
print(f"Generalization Error: {generalization_error:.6f}")
print_memory_usage("After Training")
