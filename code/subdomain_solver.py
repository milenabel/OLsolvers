import os
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells
from ufl import dx, grad, inner, div


# Ensure the "results" directory exists
results_dir = "../results/subdomain"
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



def subdomain_solver(global_solution, subdomain_bounds, refinement_factor):
    """Solve the Poisson equation on a subdomain using the global solution as a boundary condition."""
    
    # Define subdomain mesh
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

    # Define problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Compute forcing function f (same as global problem)
    f = -div(grad(ufl.sin(ufl.pi * x[0] / domain_size[0]) * ufl.sin(ufl.pi * x[1] / domain_size[1])))

    # Define a function for interpolating the global solution on the subdomain boundary
    global_solution_on_subdomain = fem.Function(V)

    # Create a bounding box tree for the global mesh
    tree = BoundingBoxTree(global_solution.function_space.mesh)

    def interpolate_global_solution(x):
        """Interpolate global solution onto the subdomain."""
        num_points = x.shape[1]
        values = np.zeros((1, num_points))  # Ensure shape (1, N)

        # Extend to 3D by padding z with zeros
        padded_x = np.vstack([x, np.zeros(num_points)])  # Convert (2, N) -> (3, N)

        # Compute colliding cells
        colliding_cells = compute_colliding_cells(tree, padded_x.T)
        cell_indices = compute_colliding_cells(global_solution.function_space.mesh, colliding_cells, padded_x.T)

        # Ensure we only evaluate where we have valid cells
        valid_mask = cell_indices >= 0
        if np.any(valid_mask):
            values[:, valid_mask] = global_solution.eval(padded_x[:, valid_mask], cell_indices[valid_mask])

        return values  # Returns a NumPy array (not a tuple)

    # Convert function into an explicit function evaluation before interpolation
    expr = fem.Expression(lambda x: interpolate_global_solution(x), V.element.interpolation_points())
    
    global_solution_on_subdomain.interpolate(expr)

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

    # Save the solution to XDMF file
    subdomain_filename = os.path.join(results_dir, f"subdomain_solution_{num_elements[0]}x{num_elements[1]}.xdmf")
    with io.XDMFFile(MPI.COMM_WORLD, subdomain_filename, "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)

    print(f"Subdomain solution saved to {subdomain_filename}")

    return msh, uh


if __name__ == "__main__":
    # Run global solver
    global_mesh, global_solution, _, _, _ = poisson_solver(domain_size=(12.0, 12.0), num_elements=(12, 12))

    # Define subdomain bounds (Example: solving only on the (0,0) to (3,3) subdomain)
    subdomain_bounds = (0, 3, 0, 3)  # (x_min, x_max, y_min, y_max)
    
    # Solve subdomain problem with refinement factor (e.g., 4x finer than global)
    sub_mesh, sub_solution = subdomain_solver(global_solution, subdomain_bounds, refinement_factor=4)

    print("Subdomain solution computed and stored successfully.")
