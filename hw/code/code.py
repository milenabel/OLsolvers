import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

figs_dir = "../figs"
os.makedirs(figs_dir, exist_ok=True)

# Define file paths (update with actual paths)
file_0_path = "../data/cleanExo_0.mat"  
file_1_path = "../data/cleanExo_1.mat"  

def load_hdf5_mat(file_path):
    """Load and inspect a MATLAB v7.3 .mat file using h5py."""
    with h5py.File(file_path, 'r') as f:
        print(f"\nLoaded file: {file_path}")
        print(f"Keys: {list(f.keys())}\n")

        # Print shape of each dataset and summary statistics
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Dataset):
                try:
                    array_data = data[()]
                    print(f"Variable '{key}': shape {array_data.shape}, dtype {array_data.dtype}")
                except:
                    print(f"Variable '{key}' is not directly readable.\n")

# Load files
load_hdf5_mat(file_0_path)
load_hdf5_mat(file_1_path)

# Load coordinate and variable data
# with h5py.File(file_0_path, 'r') as f0, h5py.File(file_1_path, 'r') as f1:
#     # Extract coordinate data
#     x0 = f0['x0'][()].flatten() if 'x0' in f0 else None
#     y0 = f0['y0'][()].flatten() if 'y0' in f0 else None
    
#     # Check node mapping if available
#     node_map = f0['node_num_map'][()] if 'node_num_map' in f0 else None

#     # Extract data for verification
#     nvar01 = f1['nvar01'][()] if 'nvar01' in f1 else None

#     print(f"\nUnique x0 values: {np.unique(x0) if x0 is not None else 'Not Found'}")
#     print(f"Unique y0 values: {np.unique(y0) if y0 is not None else 'Not Found'}")

#     if node_map is not None:
#         print(f"\nnode_num_map: {node_map}")

#     if nvar01 is not None:
#         print(f"\nnvar01 shape: {nvar01.shape} (Expected to match {x0.shape[0]} points)")

# # Scatter plot to check data mapping
# plt.figure(figsize=(6,6))
# if x0 is not None and y0 is not None:
#     plt.scatter(x0, y0, s=50, label="Original Points", c='blue', alpha=0.7)
# if nvar01 is not None:
#     plt.scatter(x0[:nvar01.shape[1]], y0[:nvar01.shape[1]], s=100, marker="o", label="nvarXX Points", c='green')

# plt.xlabel("x-coordinate")
# plt.ylabel("y-coordinate")
# plt.title("Rechecking Spatial Mapping")
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(figs_dir, "datapoints.png")) 
# plt.show()

# Load data from cleanExo_0.mat and cleanExo_1.mat
with h5py.File(file_0_path, 'r') as f0, h5py.File(file_1_path, 'r') as f1:
    
    # Extract coordinates from cleanExo_0.mat
    x0_0 = f0['x0'][()].flatten() if 'x0' in f0 else None
    y0_0 = f0['y0'][()].flatten() if 'y0' in f0 else None
    
    # Extract coordinates from cleanExo_1.mat
    x0_1 = f1['x0'][()].flatten() if 'x0' in f1 else None
    y0_1 = f1['y0'][()].flatten() if 'y0' in f1 else None

    # Extract node mapping from both files
    node_map_0 = f0['node_num_map'][()] if 'node_num_map' in f0 else None
    node_map_1 = f1['node_num_map'][()] if 'node_num_map' in f1 else None

    # Print the findings
    print("\n--- Unique Spatial Coordinates in cleanExo_0.mat ---")
    print(f"x0 (cleanExo_0): {np.unique(x0_0) if x0_0 is not None else 'Not Found'}")
    print(f"y0 (cleanExo_0): {np.unique(y0_0) if y0_0 is not None else 'Not Found'}")

    print("\n--- Unique Spatial Coordinates in cleanExo_1.mat ---")
    print(f"x0 (cleanExo_1): {np.unique(x0_1) if x0_1 is not None else 'Not Found'}")
    print(f"y0 (cleanExo_1): {np.unique(y0_1) if y0_1 is not None else 'Not Found'}")

    print("\n--- Node Number Mapping ---")
    print(f"node_num_map (cleanExo_0): {node_map_0 if node_map_0 is not None else 'Not Found'}")
    print(f"node_num_map (cleanExo_1): {node_map_1 if node_map_1 is not None else 'Not Found'}")