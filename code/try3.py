import os
import json
import numpy as np
from scipy.io import savemat

# === SETTINGS ===
mesh_sizes = [(20, 20)]
seeds = [7, 42, 77]
solution_type = "fem"  # change to "fem" ro "manu" if needed

split_base = "../results/splits"
mat_export_dir = "../results/matlab_exports"

for mesh_size in mesh_sizes:
    Nx, Ny = mesh_size
    split_dir = os.path.join(split_base, f"{Nx}x{Ny}")
    export_dir = os.path.join(mat_export_dir, f"{Nx}x{Ny}")
    os.makedirs(export_dir, exist_ok=True)

    for seed in seeds:
        print(f"Processing mesh {Nx}x{Ny}, seed {seed}...")

        # Load train data
        train_file = os.path.join(split_dir, f"{solution_type}_train_seed{seed}.json")
        test_file = os.path.join(split_dir, f"{solution_type}_test_seed{seed}.json")

        with open(train_file, "r") as f:
            train_data = json.load(f)
        with open(test_file, "r") as f:
            test_data = json.load(f)

        f_train = np.array(train_data["f_values"], dtype=np.float32)
        u_train = np.array(train_data["u_values"], dtype=np.float32)
        f_test = np.array(test_data["f_values"], dtype=np.float32)
        u_test = np.array(test_data["u_values"], dtype=np.float32)

        # Save as .mat file
        mat_data = {
            "f_train": f_train,
            "u_train": u_train,
            "f_test": f_test,
            "u_test": u_test,
        }
        save_path = os.path.join(export_dir, f"data_{solution_type}_seed{seed}.mat")
        savemat(save_path, mat_data)

        print(f"Saved: {save_path}")


# import os
# import json
# import numpy as np
# from scipy.io import savemat

# # === SETTINGS ===
# mesh_sizes = [(20, 20), (24, 24), (48, 48), (96, 96)]
# dataset_dir = "../results/general"
# export_dir = "../results/matlab_exports_full"
# os.makedirs(export_dir, exist_ok=True)

# for Nx, Ny in mesh_sizes:
#     mesh_str = f"{Nx}x{Ny}"
#     json_path = os.path.join(dataset_dir, f"poisson_dataset_{mesh_str}.json")
#     mat_path = os.path.join(export_dir, f"deeponet_full_{mesh_str}.mat")

#     print(f"Processing {json_path}...")

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     f_all = np.array(data["f_values"], dtype=np.float32)
#     u_all = np.array(data["u_values"], dtype=np.float32)

#     savemat(mat_path, {
#         "f_all": f_all,
#         "u_all": u_all
#     })

#     print(f"Saved full dataset to {mat_path}")
