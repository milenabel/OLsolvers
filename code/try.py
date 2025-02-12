from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner

def poisson_solver(domain_size=(12.0, 12.0), num_elements=(12, 12)):
    """Generalized Poisson solver that supports variable domain size and grid resolution."""
    # Create mesh
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), domain_size),
        n=num_elements,
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define problem: Poisson equation ∇⋅(k∇p) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Exact solution
    p_exact_expr = ufl.sin(ufl.pi * x[0] / domain_size[0]) * ufl.sin(ufl.pi * x[1] / domain_size[1])
    p_exact = fem.Function(V)
    p_exact.interpolate(lambda x: np.sin(np.pi * x[0] / domain_size[0]) * np.sin(np.pi * x[1] / domain_size[1]))

    # Compute f from exact solution (forcing term)
    k = ufl.sin(4 * ufl.pi * x[0]) * ufl.cos(4 * ufl.pi * x[1])
    f = -ufl.div(k * grad(p_exact_expr))

    # Define variational form
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # Solve
    problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Compute L2 error norm
    error_L2 = fem.form(inner(uh - p_exact, uh - p_exact) * dx)
    L2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2), op=MPI.SUM))

    # Compute H1 error norm
    error_H1 = fem.form(inner(grad(uh - p_exact), grad(uh - p_exact)) * dx)
    H1_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_H1), op=MPI.SUM))

    print(f"L2 error norm: {L2_norm}")
    print(f"H1 error norm: {H1_norm}")

    return msh, uh, L2_norm, H1_norm

if __name__ == "__main__":
    poisson_solver()
