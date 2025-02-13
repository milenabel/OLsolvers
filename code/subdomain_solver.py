import os
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, div

# Ensure the results directory exists
results_dir_1 = "../results/general"
os.makedirs(results_dir_1, exist_ok=True)

results_dir_2 = "../results/subdomain"
os.makedirs(results_dir_2, exist_ok=True)

def read_global_solution(global_mesh_size):
    """Read global solution from XDMF file and return function."""
    filename = os.path.join(results_dir_1, f"poisson_solution_{global_mesh_size[0]}x{global_mesh_size[1]}.xdmf")

    # Read mesh
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        msh = file.read_mesh(name="mesh")  # Ensure correct mesh name
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # Create function space
    V = fem.functionspace(msh, ("Lagrange", 1))
    uh = fem.Function(V)

    # Read function using correct format
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        file.read_mesh(msh)  # Ensure mesh is loaded first
        file.read_function(uh)  # Read function without specifying time step

    return msh, V, uh


def interpolate_global_to_subdomain(global_function, subdomain_bounds):
    """Interpolate the global function values onto the subdomain boundary."""
    def global_eval(x):
        return global_function.eval(x.T).flatten()
    
    return global_eval

def subdomain_solver(global_mesh_size, subdomain_bounds, refinement_factor):
    """Solve the Poisson equation on a subdomain using the global solution as a boundary condition."""

    # Read the global solution
    global_msh, global_V, global_uh = read_global_solution(global_mesh_size)

    # Define subdomain mesh (refinement_factor times finer than the global mesh)
    domain_size = (subdomain_bounds[1] - subdomain_bounds[0], subdomain_bounds[3] - subdomain_bounds[2])
    num_elements = (refinement_factor * (subdomain_bounds[1] - subdomain_bounds[0]),
                    refinement_factor * (subdomain_bounds[3] - subdomain_bounds[2]))

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((subdomain_bounds[0], subdomain_bounds[2]), (subdomain_bounds[1], subdomain_bounds[3])),
        n=num_elements,
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define problem: Poisson equation -Δu = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Compute f(x,y) as -Δ p_exact
    f = -div(grad(ufl.sin(ufl.pi * x[0] / domain_size[0]) * ufl.sin(ufl.pi * x[1] / domain_size[1])))

    # Interpolate global solution onto the subdomain boundary
    global_solution_on_subdomain = fem.Function(V)
    global_solution_on_subdomain.interpolate(interpolate_global_to_subdomain(global_uh, subdomain_bounds))

    # Identify subdomain boundary facets
    facets = mesh.locate_entities_boundary(msh, dim=msh.topology.dim - 1, marker=lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=facets)

    # Apply Dirichlet BC using global solution
    bc = fem.dirichletbc(global_solution_on_subdomain, dofs)

    # Define weak form
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # Solve the problem
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Save the subdomain solution
    solution_filename = os.path.join(results_dir_2, f"subdomain_solution_{num_elements[0]}x{num_elements[1]}.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, solution_filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)

    print(f"Subdomain solution saved to {solution_filename}")

    return msh, uh

if __name__ == "__main__":
    global_mesh_size = (12, 12)  # Global mesh size that we previously solved
    subdomain_bounds = (0, 3, 0, 3)  # Define subdomain bounds (x_min, x_max, y_min, y_max)
    refinement_factor = 4  # Refine subdomain mesh 4x finer than the global mesh

    print(f"Running subdomain solver for bounds {subdomain_bounds} with refinement factor {refinement_factor}...")
    sub_mesh, sub_solution = subdomain_solver(global_mesh_size, subdomain_bounds, refinement_factor)
    print("Subdomain solution computed successfully.")
