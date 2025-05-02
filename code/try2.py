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
from deeponet import DeepONet

solution_type = "FEM"
lambda_taylor_list = [0.1, 0.01, 0.001]
mesh_sizes = [(20, 20)]
split_point = 0.75
seeds = [7, 42, 77]

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    print(f"[{tag}] Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")

def compute_fourth_derivatives(model, params, x, a):
    def apply_fn(xi, ai):
        assert xi.ndim == 1 and ai.ndim == 1
        return model.apply(params, xi, ai)

    def fourth_deriv_scalar(xi, ai):
        y_fixed = jax.lax.stop_gradient(xi[1])
        x_fixed = jax.lax.stop_gradient(xi[0])
        def u_x(x_val): return apply_fn(jnp.array([x_val, y_fixed]), ai)
        def u_y(y_val): return apply_fn(jnp.array([x_fixed, y_val]), ai)
        return grad(grad(grad(grad(u_x))))(xi[0]), grad(grad(grad(grad(u_y))))(xi[1])

    return jax.vmap(fourth_deriv_scalar)(x, a)

for lambda_taylor in lambda_taylor_list:
    seed_results = []

    for seed in seeds:
        print(f"\n== λ={lambda_taylor}, seed={seed} ==")
        pyrandom = np.random.default_rng(seed)

        for mesh_size in mesh_sizes:   
            print(f"\n=== Running DeepONet for mesh {mesh_size[0]}x{mesh_size[1]} ({solution_type}) ===")         

            if solution_type == "FEM":
                with open(f"../results/general/poisson_dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
                    data = json.load(f)
                u_values = np.array(data["u_values"], dtype=np.float32)
                a_samples = np.array(data["f_values"], dtype=np.float32)
                Lx, Ly = data["domain_size"]
                with open(f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json", "r") as f:
                    u_exact_all = np.array([s["u_values"] for s in json.load(f)], dtype=np.float32)
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

            x_grid = np.linspace(0, Lx, Nx)
            y_grid = np.linspace(0, Ly, Nx)
            X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
            x_inputs = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
            x_inputs = np.tile(x_inputs[None, :, :], (num_samples, 1, 1))

            all_idx = np.arange(num_samples)
            trainval_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=seed)
            train_idx, heldout_idx = train_test_split(trainval_idx, test_size=0.125, random_state=seed)

            a_train, a_test, a_heldout = a_samples[train_idx], a_samples[test_idx], a_samples[heldout_idx]
            u_train, u_test, u_heldout = u_values[train_idx], u_values[test_idx], u_values[heldout_idx]
            x_train, x_test, x_heldout = x_inputs[train_idx], x_inputs[test_idx], x_inputs[heldout_idx]
            u_exact_heldout = u_exact_all[heldout_idx]

            print_memory_usage("After Loading Dataset")

            model = DeepONet(3, 3, 64, 1)
            key = random.PRNGKey(seed)
            params = model.init(key, jnp.array(x_train[0, 0]), jnp.array(a_train[0]))
            optimizer = optax.adam(learning_rate=0.001)
            opt_state = optimizer.init(params)

            def loss_deeponly(p, x, a, true_u):
                return jnp.mean((model.apply(p, x, a) - true_u) ** 2)

            def loss_with_taylor(p, x, a, true_u):
                pred_u = model.apply(p, x, a)
                mse_u = jnp.mean((pred_u - true_u) ** 2)
                B, G, _ = x.shape
                d4x, d4y = compute_fourth_derivatives(model, p, x.reshape(-1, 2), jnp.repeat(a, G, axis=0))
                taylor_loss = jnp.mean((d4x + d4y) ** 2)
                return mse_u + lambda_taylor * ((h1**2) / 12.0) * taylor_loss

            def make_train_step(loss_fn):
                @jax.jit
                def step(p, opt_state, x, a, true_u):
                    loss, grads = jax.value_and_grad(loss_fn)(p, x, a, true_u)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    p = optax.apply_updates(p, updates)
                    return p, opt_state, loss
                return step

            step_deeponly = make_train_step(loss_deeponly)
            step_taylor = make_train_step(loss_with_taylor)

            batch_size = 32 if grid_size <= 500 else 16 if grid_size <= 2500 else 8
            num_epochs = 100000
            switch_epoch = int(num_epochs * split_point)
            results = []

            for epoch in trange(num_epochs, desc=f"Mesh {mesh_size}, λ={lambda_taylor}, seed={seed}"):
                for b in range(0, len(x_train), batch_size):
                    x_batch = jnp.array(x_train[b:b+batch_size])
                    a_batch = jnp.array(a_train[b:b+batch_size])
                    u_batch = jnp.array(u_train[b:b+batch_size])
                    step_fn = step_deeponly if epoch < switch_epoch else step_taylor
                    params, opt_state, loss = step_fn(params, opt_state, x_batch, a_batch, u_batch)

                if epoch % 100 == 0:
                    def predict(x, a):
                        return jnp.concatenate([
                            model.apply(params, jnp.array(x[i:i+16]), jnp.array(a[i:i+16]))
                            for i in range(0, len(x), 16)
                        ], axis=0)

                    u_pred_train = predict(x_train, a_train)
                    u_pred_test = predict(x_test, a_test)
                    u_pred_heldout = predict(x_heldout, a_heldout)

                    results.append({
                        "epoch": epoch,
                        "L2_error_train": float(jnp.linalg.norm(u_pred_train - u_train) / jnp.linalg.norm(u_train)),
                        "L2_error_test": float(jnp.linalg.norm(u_pred_test - u_test) / jnp.linalg.norm(u_test)),
                        "generalization_error_fem": float(jnp.linalg.norm(u_pred_heldout - u_heldout) / jnp.linalg.norm(u_heldout)),
                        "generalization_error_true": float(jnp.linalg.norm(u_pred_heldout - u_exact_heldout) / jnp.linalg.norm(u_exact_heldout))
                    })

            out_dir = "../results/deeponets/taylor"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"deeponet_results_{solution_type}_lambda{lambda_taylor}_{mesh_size[0]}x{mesh_size[1]}_seed{seed}.json")
            with open(out_path, "w") as f:
                json.dump({"training_results": results, "seed": seed}, f)
            print(f"[✓] Saved {out_path}")
            seed_results.append(results)

    # Averaging results across seeds
    print(f"== Averaging results for λ={lambda_taylor} ==")
    avg_results = []
    for i in range(len(seed_results[0])):
        avg_entry = {
            "epoch": seed_results[0][i]["epoch"]
        }
        for key in ["L2_error_train", "L2_error_test", "generalization_error_fem", "generalization_error_true"]:
            avg_entry[key] = float(np.mean([r[i][key] for r in seed_results]))
        avg_results.append(avg_entry)

    avg_path = os.path.join(out_dir, f"deeponet_results_{solution_type}_lambda{lambda_taylor}_{mesh_size[0]}x{mesh_size[1]}_avg.json")
    with open(avg_path, "w") as f:
        json.dump({"training_results": avg_results, "seeds": seeds}, f)
    print(f"Saved averaged results to {avg_path}")
