import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import json
import os
import psutil
import time
from tqdm import trange
from jax import random, grad
from sklearn.model_selection import train_test_split
import random as pyrandom
from deeponet import DeepONet

# Choose between "manufactured" and "FEM"
solution_type = "FEM"
# lambda_taylor = 1.0
lambda_taylor_list = [1.0, 0.1, 10.0]  
mesh_sizes = [(20, 20), (24, 24)]

# Function to check memory usage
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

def compute_fourth_derivatives(model, params, x, a):
    def apply_fn(xi, ai):
        assert xi.ndim == 1 and ai.ndim == 1, f"Expected xi (2,), ai (m,), got {xi.shape}, {ai.shape}"
        return model.apply(params, xi, ai)

    def fourth_deriv_scalar(xi, ai):
        y_fixed = jax.lax.stop_gradient(xi[1])
        x_fixed = jax.lax.stop_gradient(xi[0])
        def u_x(x_val): return apply_fn(jnp.array([x_val, y_fixed]), ai)
        def u_y(y_val): return apply_fn(jnp.array([x_fixed, y_val]), ai)
        d4x = grad(grad(grad(grad(u_x))))(xi[0])
        d4y = grad(grad(grad(grad(u_y))))(xi[1])
        return d4x, d4y

    return jax.vmap(fourth_deriv_scalar)(x, a)

for lambda_taylor in lambda_taylor_list:
    for mesh_size in mesh_sizes:
        print(f"\n=== Running DeepONet for mesh {mesh_size[0]}x{mesh_size[1]} ({solution_type}) ===")

        if solution_type == "FEM":
            with open(f"../results/general/poisson_dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
                data = json.load(f)
            u_values = np.array(data["u_values"], dtype=np.float32)
            a_samples = np.array(data["f_values"], dtype=np.float32)
            Lx, Ly = data["domain_size"]
            with open(f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
                u_exact_all = np.array([sample["u_values"] for sample in json.load(f)], dtype=np.float32)
        else:
            with open(f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
                dataset = json.load(f)
            u_values = np.array([d["u_values"] for d in dataset], dtype=np.float32)
            a_samples = np.array([d["a_k"] for d in dataset], dtype=np.float32)
            Lx = Ly = 12.0
            u_exact_all = u_values

        num_samples, grid_size = u_values.shape
        Nx = int(np.sqrt(grid_size))
        h1, h2 = Lx / (Nx - 1), Ly / (Nx - 1)

        x_grid = np.linspace(0, Lx, Nx, dtype=np.float32)
        y_grid = np.linspace(0, Ly, Nx, dtype=np.float32)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        x_inputs = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
        x_inputs = np.tile(x_inputs[None, :, :], (num_samples, 1, 1))

        all_indices = np.arange(num_samples)
        trainval_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        train_idx, heldout_idx = train_test_split(trainval_idx, test_size=0.125, random_state=42)

        a_train, a_test, a_heldout = a_samples[train_idx], a_samples[test_idx], a_samples[heldout_idx]
        u_train, u_test, u_heldout = u_values[train_idx], u_values[test_idx], u_values[heldout_idx]
        x_train, x_test, x_heldout = x_inputs[train_idx], x_inputs[test_idx], x_inputs[heldout_idx]
        u_exact_heldout = u_exact_all[heldout_idx]

        print_memory_usage("After Loading Dataset")

        model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=1)
        key = random.PRNGKey(42)
        params = model.init(key, jnp.array(x_train[0, 0]), jnp.array(a_train[0]))
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, x, a, true_u):
            pred_u = model.apply(params, x, a)
            mse_u = jnp.mean((pred_u - true_u) ** 2)
            B, G, _ = x.shape
            d4x, d4y = compute_fourth_derivatives(model, params, x.reshape(-1, 2), jnp.repeat(a, G, axis=0))
            taylor_loss = jnp.mean((d4x + d4y) ** 2)
            return mse_u + lambda_taylor * ((h1 ** 2) / 12.0) * taylor_loss

        @jax.jit
        def train_step(params, opt_state, x, a, true_u):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, a, true_u)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        batch_size = 32 if grid_size <= 500 else 16 if grid_size <= 2500 else 8
        num_epochs = 100000
        training_results = []

        for epoch in trange(num_epochs, desc=f"Mesh {mesh_size}, {solution_type}"):
            for b in range(0, len(x_train), batch_size):
                x_batch = jnp.array(x_train[b:b+batch_size])
                a_batch = jnp.array(a_train[b:b+batch_size])
                u_batch = jnp.array(u_train[b:b+batch_size])
                params, opt_state, loss = train_step(params, opt_state, x_batch, a_batch, u_batch)

            if epoch % 100 == 0:
                def predict(x, a):
                    return jnp.concatenate([
                        model.apply(params, jnp.array(x[i:i+16]), jnp.array(a[i:i+16]))
                        for i in range(0, len(x), 16)
                    ], axis=0)

                u_pred_train = predict(x_train, a_train)
                u_pred_test = predict(x_test, a_test)
                u_pred_heldout = predict(x_heldout, a_heldout)

                result = {
                    "epoch": epoch,
                    "L2_error_train": float(jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)),
                    "L2_error_test": float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)),
                    "generalization_error_fem": float(jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout)),
                    "generalization_error_true": float(jnp.linalg.norm(u_pred_heldout - u_exact_heldout) / jnp.linalg.norm(u_exact_heldout))
                }

                training_results.append(result)

        results_filename = f"../results/deeponets/taylor/deeponet_results_{solution_type}_lambda{lambda_taylor}_{mesh_size[0]}x{mesh_size[1]}.json"
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        with open(results_filename, "w") as f:
            json.dump({
                "training_results": training_results,
                "solution_type": solution_type,
                "lambda_taylor": lambda_taylor
            }, f)

        print(f"Saved results to {results_filename}")