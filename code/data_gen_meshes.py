import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from tqdm import trange
import json
import os

# Parameters
num_samples = 1000
n_value = 8  # Number of basis functions
Lx, Ly = 12.0, 12.0  # Domain size
# mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (20, 20), (24, 24), (48, 48), (96, 96)]
mesh_sizes = [(20, 20)]
base_path = "../results/manufactured"

# Create base directory if needed
os.makedirs(base_path, exist_ok=True)

def generate_sample(sample_idx, V, msh):
    key = np.random.RandomState(sample_idx)
    a_k = key.normal(0, 1, size=n_value)

    x = ufl.SpatialCoordinate(msh)
    u_expr = sum(a_k[k] * ufl.cos((k + 1) * np.pi * x[0] / Lx) *
                           ufl.sin((k + 1) * np.pi * x[1] / Ly) for k in range(n_value))
    f_expr = -ufl.div(ufl.grad(u_expr))

    u = fem.Function(V)
    u.interpolate(fem.Expression(u_expr, V.element.interpolation_points()))

    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points()))

    return u, f, a_k

# Loop over mesh sizes
for Nx, Ny in mesh_sizes:
    print(f"Generating dataset for mesh {Nx}x{Ny}")
    comm = MPI.COMM_WORLD
    # msh = mesh.create_rectangle(comm, [(0.0, 0.0), (Lx, Ly)], [Nx, Ny], mesh.CellType.triangle)
    msh = mesh.create_rectangle(comm, [(0.0, 0.0), (Lx, Ly)], [Nx, Ny], mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", 1))

    dataset = []
    for i in range(num_samples):
        u, f, a_k = generate_sample(i, V, msh)
        dataset.append({
            "a_k": a_k.tolist(),
            "u_values": u.x.array.tolist(),
            "f_values": f.x.array.tolist()
        })

    # Save dataset and rhs_coeffs
    mesh_id = f"{Nx}x{Ny}"
    mesh_dir = os.path.join(base_path, mesh_id)
    os.makedirs(mesh_dir, exist_ok=True)

    dataset_path = os.path.join(mesh_dir, f"dataset_{mesh_id}.json")
    rhs_path = os.path.join(mesh_dir, f"rhs_coeffs_{mesh_id}.json")

    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    print(f"[Saved] Manufactured dataset: {dataset_path}")

    a_k_list = [sample["a_k"] for sample in dataset]
    with open(rhs_path, "w") as f:
        json.dump(a_k_list, f)
    print(f"[Saved] RHS coefficients: {rhs_path}")
 