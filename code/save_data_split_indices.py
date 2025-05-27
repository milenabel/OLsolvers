# import json
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os

# # === Configuration ===
# seeds = [7, 42, 77]
# # mesh_sizes = [(20, 20), (24, 24), (48, 48), (96, 96)]

# mesh_sizes = [(20, 20)]

# for mesh_size in mesh_sizes:
#     Nx, Ny = mesh_size
#     base_manu_path = f"../results/manufactured/{Nx}x{Ny}/dataset_{Nx}x{Ny}.json"
#     base_fem_path = f"../results/general/poisson_dataset_{Nx}x{Ny}.json"
#     save_dir = f"../results/splits/{Nx}x{Ny}"
#     os.makedirs(save_dir, exist_ok=True)

#     # === Load Manufactured Data ===
#     with open(base_manu_path, "r") as f:
#         manu_data = json.load(f)
#     a_k_manu = np.array([d["a_k"] for d in manu_data], dtype=np.float32)
#     u_manu = np.array([d["u_values"] for d in manu_data], dtype=np.float32)

#     # === Load FEM Data ===
#     with open(base_fem_path, "r") as f:
#         fem_data = json.load(f)
#     fem_f = np.array(fem_data["f_values"], dtype=np.float32)
#     fem_u = np.array(fem_data["u_values"], dtype=np.float32)

#     # === Save splits for each seed ===
#     for seed in seeds:
#         np.random.seed(seed)
#         all_indices = np.arange(len(a_k_manu))
#         train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=seed)

#         # === Save index split ===
#         indices_json = {
#             "seed": seed,
#             "train_idx": train_idx.tolist(),
#             "test_idx": test_idx.tolist()
#         }
#         with open(os.path.join(save_dir, f"train_test_split_indices_seed{seed}.json"), "w") as f:
#             json.dump(indices_json, f)

#         # === Save split manufactured data ===
#         manu_train = [{"a_k": a_k_manu[i].tolist(), "u_values": u_manu[i].tolist()} for i in train_idx]
#         manu_test = [{"a_k": a_k_manu[i].tolist(), "u_values": u_manu[i].tolist()} for i in test_idx]

#         with open(os.path.join(save_dir, f"manu_train_seed{seed}.json"), "w") as f:
#             json.dump(manu_train, f)
#         with open(os.path.join(save_dir, f"manu_test_seed{seed}.json"), "w") as f:
#             json.dump(manu_test, f)

#         # === Save split FEM data ===
#         fem_train = {
#             "f_values": fem_f[train_idx].tolist(),
#             "u_values": fem_u[train_idx].tolist(),
#             "domain_size": fem_data["domain_size"]
#         }
#         fem_test = {
#             "f_values": fem_f[test_idx].tolist(),
#             "u_values": fem_u[test_idx].tolist(),
#             "domain_size": fem_data["domain_size"]
#         }

#         with open(os.path.join(save_dir, f"fem_train_seed{seed}.json"), "w") as f:
#             json.dump(fem_train, f)
#         with open(os.path.join(save_dir, f"fem_test_seed{seed}.json"), "w") as f:
#             json.dump(fem_test, f)

#         print(f"[{Nx}x{Ny}] [Seed {seed}] Saved split files to {save_dir}")


import os
import json
import numpy as np

# === Configuration ===
seeds = [7, 42, 77]
mesh_sizes = [(20, 20)]
subdomain_refinements = [
    (3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16),
    (20, 20), (24, 24), (32, 32), (48, 48), (64, 64),
    (96, 96), (128, 128), (192, 192)
]

for mesh_size in mesh_sizes:
    Nx, Ny = mesh_size
    base_split_dir = f"../results/splits/{Nx}x{Ny}"
    os.makedirs(base_split_dir, exist_ok=True)

    for sub_mesh in subdomain_refinements:
        sx, sy = sub_mesh
        subdomain_data_path = f"../results/subdomains/fem_subdomain_g{Nx}x{Ny}_s{sx}x{sy}.json"
        if not os.path.exists(subdomain_data_path):
            print(f"[Skip] Missing subdomain data: {subdomain_data_path}")
            continue

        with open(subdomain_data_path, "r") as f:
            subdomain_data = json.load(f)

        all_u = np.array(subdomain_data["u_values"], dtype=np.float32)
        coords = subdomain_data["coords"]

        for seed in seeds:
            split_idx_path = os.path.join(base_split_dir, f"train_test_split_indices_seed{seed}.json")
            if not os.path.exists(split_idx_path):
                print(f"[Missing split indices] {split_idx_path}")
                continue

            with open(split_idx_path, "r") as f:
                split_idx = json.load(f)
            train_idx = split_idx["train_idx"]
            test_idx = split_idx["test_idx"]

            # Prepare split data
            train_data = {
                "global_mesh": [Nx, Ny],
                "sub_mesh": [sx, sy],
                "coords": coords,
                "u_values": all_u[train_idx].tolist()
            }
            test_data = {
                "global_mesh": [Nx, Ny],
                "sub_mesh": [sx, sy],
                "coords": coords,
                "u_values": all_u[test_idx].tolist()
            }

            # Save files
            train_path = os.path.join(base_split_dir, f"subdomain_train_seed{seed}_smesh{sx}x{sy}.json")
            test_path = os.path.join(base_split_dir, f"subdomain_test_seed{seed}_smesh{sx}x{sy}.json")

            with open(train_path, "w") as f:
                json.dump(train_data, f)
            with open(test_path, "w") as f:
                json.dump(test_data, f)

print("Subdomain train/test splits saved.")
