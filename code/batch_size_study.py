import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import json
import os
import matplotlib.pyplot as plt
from tqdm import trange
from jax import random
from sklearn.model_selection import train_test_split

# === DeepONet Definition ===
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

    def __call__(self, x, a):
        trunk_out = jax.vmap(self.trunk_net)(x)
        branch_out = jax.vmap(self.branch_net)(a)
        return jnp.sum(trunk_out * branch_out[:, None, :], axis=-1)

# === Training Function ===
def run_batch_size_study(mesh_size, batch_size, num_epochs=10000):
    # Load FEM dataset
    path = f"../results/general/poisson_dataset_{mesh_size[0]}x{mesh_size[1]}.json"
    with open(path, "r") as f:
        data = json.load(f)

    u_values = jnp.array(data["u_values"], dtype=jnp.float32)
    f_values = jnp.array(data["f_values"], dtype=jnp.float32)
    num_samples, grid_size = u_values.shape
    Lx, Ly = data["domain_size"]

    a_samples = f_values
    x_grid = jnp.linspace(0, Lx, int(np.sqrt(grid_size)), dtype=jnp.float32)
    y_grid = jnp.linspace(0, Ly, int(np.sqrt(grid_size)), dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    x_inputs = jnp.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    x_inputs = jnp.tile(x_inputs[None, :, :], (num_samples, 1, 1))

    # Load manufactured dataset for true generalization error
    manu_path = f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json"
    with open(manu_path, "r") as f:
        manufactured_data = json.load(f)

    u_exact_all = jnp.array([sample["u_values"] for sample in manufactured_data], dtype=jnp.float32)
    a_samples = jnp.array([sample["a_k"] for sample in manufactured_data], dtype=jnp.float32)

    # Split datasets
    a_train, a_test, u_train, u_test, x_train, x_test = train_test_split(a_samples, u_values, x_inputs, test_size=0.2, random_state=42)
    a_train, a_heldout, u_train, u_heldout, x_train, x_heldout = train_test_split(a_train, u_train, x_train, test_size=0.125, random_state=42)

    _, _, u_exact_train, u_exact_heldout = train_test_split(a_samples, u_exact_all, test_size=0.2, random_state=42)
    _, u_exact_heldout = train_test_split(u_exact_train, test_size=0.125, random_state=42)

    # Initialize model
    model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
    key = random.PRNGKey(42)
    params = model.init(key, x_train[0:1], a_train[0:1])
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, a, true_u):
        pred_u = model.apply(params, x, a)
        return jnp.mean((pred_u - true_u) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, a, true_u):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, a, true_u)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    results = []
    num_batches = max(1, len(x_train) // batch_size)

    for epoch in trange(num_epochs, desc=f"Mesh {mesh_size}, Batch {batch_size}"):
        for b in range(num_batches):
            i = b * batch_size
            x_batch = x_train[i:i+batch_size]
            a_batch = a_train[i:i+batch_size]
            u_batch = u_train[i:i+batch_size]
            params, opt_state, _ = train_step(params, opt_state, x_batch, a_batch, u_batch)

        if epoch % 100 == 0:
            def predict_in_batches(x, a):
                preds = []
                for i in range(0, len(x), 16):
                    preds.append(model.apply(params, x[i:i+16], a[i:i+16]))
                return jnp.concatenate(preds, axis=0)

            u_pred_train = predict_in_batches(x_train, a_train)
            u_pred_test = predict_in_batches(x_test, a_test)
            u_pred_heldout = predict_in_batches(x_heldout, a_heldout)

            L2_train = jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)
            L2_test = jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)
            L2_gen_fem = jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout)
            L2_gen_true = jnp.linalg.norm(u_pred_heldout - u_exact_heldout) / jnp.linalg.norm(u_exact_heldout)

            results.append({
                "epoch": epoch,
                "L2_error_train": float(L2_train),
                "L2_error_test": float(L2_test),
                "generalization_error_fem": float(L2_gen_fem),
                "generalization_error_true": float(L2_gen_true)
            })

    save_path = f"../results/batch_study/deeponet_results_FEM_{mesh_size[0]}x{mesh_size[1]}_batch{batch_size}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"training_results": results}, f)
    print(f"Saved: {save_path}")

# === Run All Experiments ===
mesh_sizes = [(20, 20), (24, 24), (48, 48)]
batch_sizes = [8, 16, 32, 64, 128]

for mesh in mesh_sizes:
    for b in batch_sizes:
        run_batch_size_study(mesh_size=mesh, batch_size=b)

# === Plotting ===
    train_errors = []
    test_errors = []
    gen_errors_fem = []
    gen_errors_true = []

    for b in batch_sizes:
        path = f"../results/batch_study/deeponet_results_FEM_{mesh[0]}x{mesh[1]}_batch{b}.json"
        with open(path, "r") as f:
            res = json.load(f)
            last = res["training_results"][-1]
            train_errors.append(last["L2_error_train"])
            test_errors.append(last["L2_error_test"])
            gen_errors_fem.append(last["generalization_error_fem"])
            gen_errors_true.append(last["generalization_error_true"])

    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, train_errors, 'o-', label="Train L2 Error")
    plt.plot(batch_sizes, test_errors, 's-', label="Test L2 Error")
    plt.plot(batch_sizes, gen_errors_fem, '^-', label="Gen Error vs FEM")
    plt.plot(batch_sizes, gen_errors_true, 'v-', label="Gen Error vs Manufactured")
    plt.xscale("log", base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Effect of Batch Size on Accuracy (Mesh {mesh[0]}x{mesh[1]})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = f"../figs/batch_size_study_{mesh[0]}x{mesh[1]}.png"
    plt.savefig(fig_name)
    print(f"Saved plot to {fig_name}")
