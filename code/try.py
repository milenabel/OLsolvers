import os
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import random, grad, value_and_grad
from tqdm import trange
from deeponet import DeepONet  
import time
import psutil

# === SETTINGS ===
solution_type = "FEM"
mesh_sizes = [(20, 20)]
lambda_taylor_list = [0.1, 0.01, 0.001]
lambda_phys_list = [1, 0.1, 0.01, 0.001]
lambda_reg_list = [1e-6]
seeds = [7, 42, 77]
num_epochs = 100000

# === MEMORY CHECK UTILITY ===
def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

# === FOURTH ORDER DERIVATIVE UTILITY ===
def compute_fourth_derivatives(model, params, x, a):
    def apply_fn(xi, ai):
        return model.apply(params, xi, ai)

    def fourth_deriv_scalar(xi, ai):
        x_fixed, y_fixed = xi[0], xi[1]

        def u_x(x_val): return apply_fn(jnp.array([x_val, y_fixed]), ai)
        def u_y(y_val): return apply_fn(jnp.array([x_fixed, y_val]), ai)

        d4x = grad(grad(grad(grad(u_x))))(x_fixed)
        d4y = grad(grad(grad(grad(u_y))))(y_fixed)
        return d4x, d4y

    return jax.vmap(fourth_deriv_scalar)(x, a)

# === PHYSICS RESIDUAL ===
def compute_laplacian(model, params, x, a):
    def u_fn(xi, ai):
        return model.apply(params, xi, ai)

    def laplace_scalar(xi, ai):
        def u_x(x_): return u_fn(jnp.array([x_, xi[1]]), ai)
        def u_y(y_): return u_fn(jnp.array([xi[0], y_]), ai)

        d2x = grad(grad(u_x))(xi[0])
        d2y = grad(grad(u_y))(xi[1])
        return d2x + d2y

    return jax.vmap(laplace_scalar)(x, a)

# === TRAINING FUNCTION ===
def run_training_for_seed(seed, mesh_size, lambda_taylor, lambda_phys, lambda_reg):
    split_dir = f"../results/splits/{mesh_size[0]}x{mesh_size[1]}"
    data_dir = "../results/general"

    # === Load split data ===
    with open(os.path.join(split_dir, f"fem_train_seed{seed}.json")) as f:
        fem_train = json.load(f)
    with open(os.path.join(split_dir, f"fem_test_seed{seed}.json")) as f:
        fem_test = json.load(f)

    a_train = jnp.array(fem_train["f_values"], dtype=jnp.float32)
    u_train = jnp.array(fem_train["u_values"], dtype=jnp.float32)
    a_test = jnp.array(fem_test["f_values"], dtype=jnp.float32)
    u_test = jnp.array(fem_test["u_values"], dtype=jnp.float32)
    Lx, Ly = fem_train["domain_size"]

    # === Load manufactured truth ===
    with open(f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
        manu_data = json.load(f)
    u_manu_test = jnp.array([d["u_values"] for d in manu_data], dtype=jnp.float32)[
        np.array(json.load(open(os.path.join(split_dir, f"train_test_split_indices_seed{seed}.json")))["test_idx"])
    ]

    # === Prepare grid ===
    grid_size = u_train.shape[1]
    Nx = int(np.sqrt(grid_size))
    h = Lx / (Nx - 1)
    x_vals = jnp.linspace(0, Lx, Nx, dtype=jnp.float32)
    y_vals = jnp.linspace(0, Ly, Nx, dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing="ij")
    x_inputs = jnp.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    x_train = jnp.tile(x_inputs[None, :, :], (u_train.shape[0], 1, 1))
    x_test = jnp.tile(x_inputs[None, :, :], (u_test.shape[0], 1, 1))

    # === Initialize model ===
    model = DeepONet(trunk_layers=3, branch_layers=3, hidden_dim=64, output_dim=grid_size)
    key = random.PRNGKey(seed)
    params = model.init(key, x_train[0:1], a_train[0:1])
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # === Composite Loss ===
    def composite_loss(params, x, a, true_u, true_f):
        pred_u = model.apply(params, x, a)
        mse_term = jnp.mean((pred_u - true_u) ** 2)

        # Taylor term
        B, G, _ = x.shape
        d4x, d4y = compute_fourth_derivatives(model, params, x.reshape(-1, 2), jnp.repeat(a, G, axis=0))
        taylor_term = jnp.mean((d4x + d4y) ** 2)

        # Physics residual term
        lap_u = compute_laplacian(model, params, x.reshape(-1, 2), jnp.repeat(a, G, axis=0)).reshape(B, G)
        f_flat = true_f.reshape(B, G)
        physics_term = jnp.mean((lap_u - f_flat) ** 2)

        # Weight regularization
        weight_sq_sum = sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)])
        reg_term = lambda_reg * weight_sq_sum

        return mse_term + lambda_taylor * (h ** 2 / 12.0) * taylor_term + lambda_phys * physics_term + reg_term

    @jax.jit
    def train_step(params, opt_state, x, a, true_u, true_f):
        loss, grads = value_and_grad(composite_loss)(params, x, a, true_u, true_f)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def predict_fn(params, x, a):
        return model.apply(params, x, a)

    def predict_in_batches(x, a, batch_size=16):
        return jnp.concatenate([predict_fn(params, x[i:i+batch_size], a[i:i+batch_size])
                                for i in range(0, len(x), batch_size)], axis=0)

    # === Train Loop ===
    training_results = []
    if grid_size <= 500:
        batch_size = 64
    elif grid_size <= 2500:
        batch_size = 32
    else:
        batch_size = 8
    num_batches = max(1, len(x_train) // batch_size)

    for epoch in trange(num_epochs, desc=f"Training DeepONet FEM MSE+Taylor+Physics+Regularization - {mesh_size} - Seed {seed} - λ_1={lambda_taylor} - λ_2={lambda_phys}"):
        epoch_start = time.time()
        for batch in range(num_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size
            x_batch = x_train[start:end]
            a_batch = a_train[start:end]
            u_batch = u_train[start:end]
            f_batch = a_train[start:end]  # f = a for FEM inputs
            params, opt_state, _ = train_step(params, opt_state, x_batch, a_batch, u_batch, f_batch)

        if epoch % 100 == 0:
            u_pred_train = predict_in_batches(x_train, a_train)
            L2_error_train = float(jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train))
            training_results.append({
                "epoch": epoch,
                "L2_error_train": L2_error_train,
            })

            print_memory_usage(f"Epoch {epoch}")
            print(f"[Epoch {epoch}] λ_1={lambda_taylor} - λ_2={lambda_phys}, L2 Error Train = {L2_error_train:.6f}")
            print(f"Epoch time: {time.time() - epoch_start:.2f}s")

    # === Final Evaluation ===
    u_pred_test = predict_in_batches(x_test, a_test)
    L2_test = float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test))
    gen_error_fem = float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test))
    gen_error_true = float(jnp.linalg.norm(u_pred_test - u_manu_test) / jnp.linalg.norm(u_manu_test))

    return {
        "seed": seed,
        "L2_error_test": L2_test,
        "generalization_error_fem": gen_error_fem,
        "generalization_error_true": gen_error_true,
    }

# === MAIN EXECUTION ===
for mesh_size in mesh_sizes:
    for lambda_taylor in lambda_taylor_list:
        for lambda_phys in lambda_phys_list:
            for lambda_reg in lambda_reg_list:
                results_dir = f"../results/deeponets/tayph/{mesh_size[0]}x{mesh_size[1]}"
                os.makedirs(results_dir, exist_ok=True)

                all_results = []
                for seed in seeds:
                    result = run_training_for_seed(seed, mesh_size, lambda_taylor, lambda_phys, lambda_reg)
                    all_results.append(result)

                    file_path = os.path.join(results_dir, f"deeponet_results_FEM_lambdaT{lambda_taylor}_lambdaP{lambda_phys}_reg{lambda_reg}_seed{seed}.json")
                    with open(file_path, "w") as f:
                        json.dump(result, f, indent=2)

                # Save average
                avg_result = {k: float(np.mean([r[k] for r in all_results]))
                    for k in all_results[0] if k != "seed"
                }
                avg_path = os.path.join(results_dir, f"deeponet_results_FEM_lambdaT{lambda_taylor}_lambdaP{lambda_phys}_reg{lambda_reg}_avg.json")
                with open(avg_path, "w") as f:
                    json.dump(avg_result, f, indent=2)

                print(f"[Saved] Averaged results -> {avg_path}")

