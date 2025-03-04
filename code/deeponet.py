import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import json
from jax import random
from sklearn.model_selection import train_test_split

# Load dataset
with open("../results/deeponets/1/dataset1000.json", "r") as f:
    dataset = json.load(f)

a_samples = jnp.array([sample["a_k"] for sample in dataset])  # Shape (num_samples, 50)
u_values = jnp.array([sample["u_values"] for sample in dataset])  # Shape (num_samples, grid_size)
f_values = jnp.array([sample["f_values"] for sample in dataset])  # Shape (num_samples, grid_size)


num_samples, grid_size = u_values.shape
Lx, Ly = 12.0, 12.0  # Domain size
x_grid = jnp.linspace(0, Lx, int(jnp.sqrt(grid_size)))  
y_grid = jnp.linspace(0, Ly, int(jnp.sqrt(grid_size)))
X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
X = X.flatten()[:, None]  # Reshape to column vector
Y = Y.flatten()[:, None]
x_inputs = jnp.hstack([X, Y])  # Shape: (grid_size, 2)

# Tile x_inputs so each sample has (grid_size, 2)
x_inputs = jnp.tile(x_inputs[None, :, :], (num_samples, 1, 1))  # Shape: (num_samples, grid_size, 2)

# Split data into training and testing sets (80/20 split)
a_train, a_test, u_train, u_test, x_train, x_test = train_test_split(
    a_samples, u_values, x_inputs, test_size=0.2, random_state=42
)

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
        trunk_out = jax.vmap(self.trunk_net)(x)  # Shape: (num_samples, grid_size, output_dim)
        branch_out = self.branch_net(a)[:, None, :]  # Shape: (num_samples, 1, output_dim)
        return jnp.sum(trunk_out * branch_out, axis=-1)  # Shape: (num_samples, grid_size)

# Initialize model
model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
key = random.PRNGKey(0)
params = model.init(key, x_train, a_train)

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

# **Storage for tracking results**
training_results = []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    params, opt_state, loss = train_step(params, opt_state, x_train, a_train, u_train)
    
    # Compute DeepONet prediction on training data
    u_pred_train = model.apply(params, x_train, a_train)

    # Compute L2 error on training data
    L2_error_train = jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)

    # Compute DeepONet prediction on testing data
    u_pred_test = model.apply(params, x_test, a_test)

    # Compute L2 error on testing data
    L2_error_test = jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)

    # Store results at each epoch
    training_results.append({
        "epoch": epoch,
        "u_pred_train": u_pred_train.tolist(),
        "u_true_train": u_train.tolist(),
        "L2_error_train": float(L2_error_train),
        "u_pred_test": u_pred_test.tolist(),
        "u_true_test": u_test.tolist(),
        "L2_error_test": float(L2_error_test)
    })

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}, L2 Error Train: {L2_error_train:.6f}, L2 Error Test: {L2_error_test:.6f}")
        print(f"  U_true_train (first 5): {u_train.flatten()[:5]}")
        print(f"  U_pred_train (first 5): {u_pred_train.flatten()[:5]}")
        print(f"  U_true_test (first 5): {u_test.flatten()[:5]}")
        print(f"  U_pred_test (first 5): {u_pred_test.flatten()[:5]}")

# Save results to JSON
with open("../results/deeponets/1/deeponet_results.json", "w") as f:
    json.dump(training_results, f)

print("Training complete. Results saved to deeponet_results.json")
 
