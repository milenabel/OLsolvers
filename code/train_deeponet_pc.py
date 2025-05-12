import jax
import jax.numpy as jnp
import numpy as np
import optax
import json
import os
from tqdm import trange
from jax import random, grad, value_and_grad
from sklearn.model_selection import train_test_split
from deeponet_pc import Predictor, Corrector, PredictorCorrector

# === Settings ===
mesh_size = (20, 20)
split_point = 0.75
num_epochs = 100000
batch_size = 32
learning_rate = 0.001
results_dir = "../results/deeponets/predictor_corrector"
os.makedirs(results_dir, exist_ok=True)

# === Load Data ===
fem_path = f"../results/general/poisson_dataset_{mesh_size[0]}x{mesh_size[1]}.json"
with open(fem_path, "r") as f:
    fem_data = json.load(f)
fem_u = np.array(fem_data["u_values"], dtype=np.float32)
fem_f = np.array(fem_data["f_values"], dtype=np.float32)

manu_path = f"../results/manufactured/{mesh_size[0]}x{mesh_size[1]}/dataset_{mesh_size[0]}x{mesh_size[1]}.json"
with open(manu_path, "r") as f:
    manu_data = json.load(f)
manu_u = np.array([sample["u_values"] for sample in manu_data], dtype=np.float32)

# === Train/Test Split ===
a_train, a_test, fem_u_train, fem_u_test, manu_u_train, manu_u_test = train_test_split(
    fem_f, fem_u, manu_u, test_size=0.2, random_state=42
)

# === Model Setup ===
key = random.PRNGKey(42)
predictor = Predictor(branch_layers=3, trunk_layers=3, hidden_dim=64, output_dim=fem_u.shape[1])
corrector = Corrector(branch_layers=3, trunk_layers=3, hidden_dim=64, output_dim=fem_u.shape[1])
model = PredictorCorrector(predictor=predictor, corrector=corrector)

x_dummy = jnp.ones((1, 2), dtype=jnp.float32)
a_dummy = jnp.ones((1, a_train.shape[1]), dtype=jnp.float32)
init_variables = model.init(key, x_dummy, a_dummy)
params = init_variables["params"]

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# === Wrapper functions ===
def call_predictor(params, x, a_input):
    return model.predictor.apply({"params": params["predictor"]}, x, a_input)

def call_corrector(params, x, u_approx):
    return model.corrector.apply({"params": params["corrector"]}, x, u_approx)

# === Loss Functions ===
def loss_predictor(params, a_input, fem_targets):
    x_dummy = jnp.ones((a_input.shape[0], 2), dtype=jnp.float32)
    pred = call_predictor(params, x_dummy, a_input)
    return jnp.mean((pred - fem_targets) ** 2)

def loss_corrector(params, a_input, manu_targets):
    pred = call_predictor(params, a_input)
    x_dummy = jnp.ones((a_input.shape[0], 2), dtype=jnp.float32)
    corrected = call_corrector(params, x_dummy, pred)
    return jnp.mean((corrected - manu_targets) ** 2)

def loss_combined(params, a_input, manu_targets):
    pred = call_predictor(params, a_input)
    x_dummy = jnp.ones((a_input.shape[0], 2), dtype=jnp.float32)
    corrected = call_corrector(params, x_dummy, pred)
    return jnp.mean((corrected - manu_targets) ** 2)

# === Training Step Generator ===
def make_train_step(loss_fn):
    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step

train_step_predictor = make_train_step(loss_predictor)
train_step_combined = make_train_step(loss_combined)

# === Training Loop ===
results = []
for epoch in trange(num_epochs, desc="Training Predictor-Corrector"):
    # === Phase Select ===
    if epoch < int(num_epochs * split_point):
        step_fn = train_step_predictor
        inputs, targets = a_train, fem_u_train
    else:
        step_fn = train_step_combined
        inputs, targets = a_train, manu_u_train

    # === Batch Training ===
    for i in range(0, len(inputs), batch_size):
        x_batch = jnp.array(inputs[i:i+batch_size])
        y_batch = jnp.array(targets[i:i+batch_size])
        params, opt_state, loss = step_fn(params, opt_state, x_batch, y_batch)

    # === Logging ===
    if epoch % 100 == 0:
        pred_test = call_predictor(params, jnp.array(a_test))
        x_test_dummy = jnp.ones((a_test.shape[0], 2), dtype=jnp.float32)
        corrected_test = call_corrector(params, x_test_dummy, pred_test)

        L2_pred = float(jnp.linalg.norm(pred_test - fem_u_test) / jnp.linalg.norm(fem_u_test))
        L2_corr = float(jnp.linalg.norm(corrected_test - manu_u_test) / jnp.linalg.norm(manu_u_test))

        results.append({
            "epoch": epoch,
            "L2_predictor_test": L2_pred,
            "L2_corrector_test": L2_corr,
        })

        print(f"[Epoch {epoch}] L2 Predictor: {L2_pred:.4e}, L2 Corrector: {L2_corr:.4e}")

# === Save Results ===
results_file = os.path.join(results_dir, f"predictor_corrector_results_{mesh_size[0]}x{mesh_size[1]}.json")
with open(results_file, "w") as f:
    json.dump(results, f)
print("Saved training results to:", results_file)