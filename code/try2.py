import json
import numpy as np
import jax.numpy as jnp

# === Settings ===
mesh = "20x20"
Lx, Ly = 12.0, 12.0
n_basis = 8

# === Load datasets ===
with open(f"../results/manufactured/{mesh}/dataset_{mesh}.json") as f:
    manu_data = json.load(f)
with open(f"../results/general/poisson_dataset_{mesh}.json") as f:
    fem_data = json.load(f)

# === Extract data ===
a_k_manu = jnp.array([d["a_k"] for d in manu_data])
f_manu = jnp.array([d["f_values"] for d in manu_data])
u_manu = jnp.array([d["u_values"] for d in manu_data])

f_fem = jnp.array(fem_data["f_values"])
u_fem = jnp.array(fem_data["u_values"])
domain_size = fem_data["domain_size"]

# === Consistency Checks ===

# 1. Check that domain matches
assert np.allclose(domain_size, [Lx, Ly]), f"Domain mismatch: {domain_size} != {Lx, Ly}"

# 2. Check that number of samples match
assert f_fem.shape == f_manu.shape, f"Shape mismatch: f_fem {f_fem.shape}, f_manu {f_manu.shape}"
assert u_fem.shape == u_manu.shape, f"u shape mismatch: {u_fem.shape} vs {u_manu.shape}"

# 3. Check that RHS values are close
f_diff = jnp.linalg.norm(f_fem - f_manu) / jnp.linalg.norm(f_manu)
print(f"Relative L2 difference in RHS (f): {f_diff:.6e}")

# 4. Check that solution values aren't wildly off
u_diff = jnp.linalg.norm(u_fem - u_manu) / jnp.linalg.norm(u_manu)
print(f"Relative L2 difference in u: {u_diff:.6e}")

# 5. Optional: Check individual sample match (first one)
sample_id = 0
print("\nSample 0:")
print(f"||f_fem - f_manu|| / ||f_manu|| = {jnp.linalg.norm(f_fem[sample_id] - f_manu[sample_id]) / jnp.linalg.norm(f_manu[sample_id]):.6e}")
print(f"||u_fem - u_manu|| / ||u_manu|| = {jnp.linalg.norm(u_fem[sample_id] - u_manu[sample_id]) / jnp.linalg.norm(u_manu[sample_id]):.6e}")
