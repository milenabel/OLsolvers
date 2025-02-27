import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, io
import json

# Set parameters
num_samples = 100  # Number of solutions to generate
n_value = 8  # Frequency factor
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 20, 20  # Mesh resolution

def generate_sample(sample_idx):
    """Generate a sample of the manufactured solution and its corresponding f."""
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [(0.0, 0.0), (Lx, Ly)], [Nx, Ny], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    # Generate random coefficients from Gaussian distribution
    key = np.random.RandomState(sample_idx)
    a_k = key.normal(0, 1, size=n_value)
    
    # Define manufactured solution u_a(x, y)
    u_expr = sum(a_k[k] * ufl.cos((k+1) * np.pi * ufl.SpatialCoordinate(msh)[0]) *
                       ufl.sin((k+1) * np.pi * ufl.SpatialCoordinate(msh)[1]) for k in range(n_value))
    
    u = fem.Function(V)
    u.interpolate(fem.Expression(u_expr, V.element.interpolation_points()))
    
    # Compute ∆u_a(x, y)
    f_expr = -ufl.div(ufl.grad(u_expr))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points()))
    
    return u, f, a_k

# Store dataset
dataset = []
for i in range(num_samples):
    u, f, a_k = generate_sample(i)
    dataset.append({
        "a_k": a_k.tolist(),
        "u_values": u.x.array.tolist(),
        "f_values": f.x.array.tolist()
    })

# Save dataset to file
with open("../results/deeponets/1/dataset.json", "w") as f:
    json.dump(dataset, f)

print(f"Saved dataset with {num_samples} samples to dataset.json")
