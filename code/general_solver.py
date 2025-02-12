import os
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, div
import pandas as pd

# Ensure the "results" directory exists
results_dir = "../results/general"
os.makedirs(results_dir, exist_ok=True)

def poisson_solver(domain_size=(12.0, 12.0), num_elements=(12, 12)):
    """Generalized Poisson solver with error computation and result storage."""
    # Create mesh
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), domain_size),
        n=num_elements,
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define problem: Poisson equation -Δu = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Define exact solution (used for computing f and errors)
    p_exact_expr = ufl.sin(ufl.pi * x[0] / domain_size[0]) * ufl.sin(ufl.pi * x[1] / domain_size[1])
    
    # Compute f(x,y) as -Δ p_exact
    f = -div(grad(p_exact_expr))

    # Interpolate exact solution onto function space
    p_exact = fem.Function(V)
    def exact_solution(x):
        return np.sin(np.pi * x[0] / domain_size[0]) * np.sin(np.pi * x[1] / domain_size[1])
    p_exact.interpolate(exact_solution)

    # Apply Dirichlet BC to match the exact solution
    facets = mesh.locate_entities_boundary(
        msh, dim=msh.topology.dim - 1, marker=lambda x: np.full(x.shape[1], True)  # Apply BCs to all boundaries
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(p_exact, dofs)  # Use exact solution as BC

    # Define weak form
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # Solve the problem
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Compute L2 and H1 error norms
    error_L2_form = fem.form(inner(uh - p_exact, uh - p_exact) * dx)
    L2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_form), op=MPI.SUM))

    error_H1_form = fem.form(inner(grad(uh - p_exact), grad(uh - p_exact)) * dx)
    H1_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_H1_form), op=MPI.SUM))

    # Compute relative L2 error norm
    p_exact_L2_form = fem.form(inner(p_exact, p_exact) * dx)
    p_exact_L2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(p_exact_L2_form), op=MPI.SUM))

    relative_L2_error = L2_norm / p_exact_L2_norm

    # Save the solution to XDMF file
    solution_filename = os.path.join(results_dir, f"poisson_solution_{num_elements[0]}x{num_elements[1]}.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, solution_filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)

    return msh, uh, L2_norm, H1_norm, relative_L2_error

if __name__ == "__main__":
    mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (24, 24), (48, 48), (96, 96)]
    
    # Store results for CSV output
    results = []

    for size in mesh_sizes:
        print(f"Running solver for mesh size: {size}")
        _, _, L2_norm, H1_norm, relative_L2_error = poisson_solver(num_elements=size)
        print(f"Mesh {size}: L2 norm = {L2_norm}, H1 norm = {H1_norm}, Relative L2 norm = {relative_L2_error}\n")

        # Store results
        results.append([size[0], size[1], L2_norm, H1_norm, relative_L2_error])

    # Save error norms to a CSV file
    results_filename = os.path.join(results_dir, "poisson_error_norms.csv")
    df = pd.DataFrame(results, columns=["Nx", "Ny", "L2 Norm", "H1 Norm", "Relative L2 Norm"])
    df.to_csv(results_filename, index=False)

    print(f"Results saved to {results_filename}")
