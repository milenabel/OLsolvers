import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm import trange
from jax import random
from deeponet import DeepONet
import time
import psutil

# === SETTINGS ===
solution_type = "manufactured"  # "manufactured" or "FEM"
mesh_sizes = [(20, 20)]
seeds = [7, 42, 77]
num_epochs = 100000

# === MEMORY CHECK UTILITY ===
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

# === TRAINING FUNCTION ===
def run_training_for_seed(seed, mesh_size, results_dir):
    # Nx, Ny = mesh_size
    split_dir = f"../results/splits/{mesh_size[0]}x{mesh_size[1]}"
    data_dir = f"../results/{'manufactured' if solution_type == 'manufactured' else 'general'}"

    # === Load split indices ===
    with open(os.path.join(split_dir, f"train_test_split_indices_seed{seed}.json")) as f:
        split_indices = json.load(f)
    train_idx = np.array(split_indices["train_idx"])
    test_idx = np.array(split_indices["test_idx"])

    # === Load corresponding data ===
    if solution_type == "manufactured":
        with open(os.path.join(split_dir, f"manu_train_seed{seed}.json")) as f:
            manu_train = json.load(f)
        with open(os.path.join(split_dir, f"manu_test_seed{seed}.json")) as f:
            manu_test = json.load(f)

        a_train = jnp.array([d["a_k"] for d in manu_train], dtype=jnp.float32)
        u_train = jnp.array([d["u_values"] for d in manu_train], dtype=jnp.float32)
        a_test = jnp.array([d["a_k"] for d in manu_test], dtype=jnp.float32)
        u_test = jnp.array([d["u_values"] for d in manu_test], dtype=jnp.float32)
        Lx, Ly = 12.0, 12.0

    elif solution_type == "FEM":
        with open(os.path.join(split_dir, f"fem_train_seed{seed}.json")) as f:
            fem_train = json.load(f)
        with open(os.path.join(split_dir, f"fem_test_seed{seed}.json")) as f:
            fem_test = json.load(f)

        a_train = jnp.array(fem_train["f_values"], dtype=jnp.float32)
        u_train = jnp.array(fem_train["u_values"], dtype=jnp.float32)
        a_test = jnp.array(fem_test["f_values"], dtype=jnp.float32)
        u_test = jnp.array(fem_test["u_values"], dtype=jnp.float32)
        Lx, Ly = fem_train["domain_size"]

    # === For generalization error vs manufactured ===
    manu_path = f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json"
    with open(manu_path, "r") as f:
        manu_data = json.load(f)
    u_manu_all = jnp.array([d["u_values"] for d in manu_data], dtype=jnp.float32)
    u_manu_test = u_manu_all[test_idx]

    # === Prepare x grid ===
    grid_size = u_train.shape[1]
    Nx = int(np.sqrt(grid_size))  # infer Nx from data
    Ny = Nx
    assert Nx * Ny == u_train.shape[1], "Grid size doesn't match expected square layout"
    x_grid = jnp.linspace(0, Lx, Nx, dtype=jnp.float32)
    y_grid = jnp.linspace(0, Ly, Ny, dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    x_inputs = jnp.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    x_train = jnp.tile(x_inputs[None, :, :], (u_train.shape[0], 1, 1))
    x_test = jnp.tile(x_inputs[None, :, :], (u_test.shape[0], 1, 1))

    # === Initialize model ===
    model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
    key = random.PRNGKey(seed)
    params = model.init(key, x_train[0:1], a_train[0:1])
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # === Loss and Train step ===
    def loss_fn(params, x, a, true_u):
        pred_u = model.apply(params, x, a)
        return jnp.mean((pred_u - true_u) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, a, true_u):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, a, true_u)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict_fn(params, x, a):
        return model.apply(params, x, a)

    def predict_in_batches(x, a, batch_size=16):
        preds = []
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            a_batch = a[i:i+batch_size]
            pred = predict_fn(params, x_batch, a_batch)
            preds.append(pred)
        return jnp.concatenate(preds, axis=0)

    # === Train ===
    training_results = []
    if grid_size <= 500:
        batch_size = 64
    elif grid_size <= 2500:
        batch_size = 32
    else:
        batch_size = 8
    num_batches = max(1, len(x_train) // batch_size)

    for epoch in trange(num_epochs, desc=f"Training DeepONet {solution_type} - {mesh_size} - Seed {seed}"):
        epoch_start = time.time()
        for batch in range(num_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size
            x_batch = x_train[start:end]
            a_batch = a_train[start:end]
            u_batch = u_train[start:end]
            params, opt_state, _ = train_step(params, opt_state, x_batch, a_batch, u_batch)

        if epoch % 1000 == 0:
            u_pred_train = predict_in_batches(x_train, a_train)

            # Compute L2 errors
            L2_error_train = jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)

            # Store results
            training_results.append({
                "epoch": epoch,
                "L2_error_train": float(L2_error_train),
            })

            print_memory_usage(f"Epoch {epoch}")
            print(f"Epoch {epoch}, L2 Error Train: {L2_error_train:.6f}")
            print(f"Epoch time: {time.time() - epoch_start:.2f}s")
        
            temp_result_dict = {
                "seed": seed,
                "training_results": training_results,
            }
            file_path = os.path.join(results_dir, f"deeponet_results_{solution_type}_{mesh_size[0]}x{mesh_size[1]}_seed{seed}.json")
            with open(file_path, "w") as f:
                json.dump(temp_result_dict, f, indent=2)



    # === Final Evaluation ===
    
    u_pred_test = predict_in_batches(x_test, a_test)

    L2_test = float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test))

    if solution_type == "FEM":
        gen_error_fem = float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test))
        gen_error_true = float(jnp.linalg.norm(u_pred_test - u_manu_test) / jnp.linalg.norm(u_manu_test))
    elif solution_type == "manufactured":
        gen_error_true = L2_test

    result_dict = {
        "seed": seed,
        "training_results": training_results,
        "L2_error_test": L2_test,
        "generalization_error_true": gen_error_true,
    }
    if solution_type == "FEM":
        result_dict["generalization_error_fem"] = gen_error_fem

    file_path = os.path.join(results_dir, f"deeponet_results_{solution_type}_{mesh_size[0]}x{mesh_size[1]}_seed{seed}.json")
    with open(file_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    return result_dict


# === MAIN EXECUTION ===
for mesh_size in mesh_sizes:
    results_dir = f"../results/deeponets/2/{mesh_size[0]}x{mesh_size[1]}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for seed in seeds:
        result = run_training_for_seed(seed, mesh_size, results_dir)
        all_results.append(result)

        # Save per-seed results
        file_path = os.path.join(results_dir, f"deeponet_results_{solution_type}_{mesh_size[0]}x{mesh_size[1]}_seed{seed}.json")
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save averaged results
    avg_result = {}

    for key in all_results[0]:
        if key in ("seed", "training_results"): continue
        avg_result[key] = float(np.mean([r[key] for r in all_results]))


    avg_file = os.path.join(results_dir, f"deeponet_results_{solution_type}_avg.json")
    with open(avg_file, "w") as f:
        json.dump(avg_result, f, indent=2)

    print(f"Saved averaged results to {avg_file}")
