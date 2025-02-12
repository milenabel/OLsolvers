from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

def create_mesh(domain_size, num_elements):
    """Create a structured quadrilateral mesh with adjustable size."""
    return mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), domain_size),
        n=num_elements,
        cell_type=mesh.CellType.quadrilateral,
    )

def poisson_solver(domain_size=(12.0, 12.0), num_elements=(12, 12)):
    """Generalized Poisson solver that supports variable domain size and grid resolution."""
    # Create mesh
    msh = create_mesh(domain_size, num_elements)
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define boundary conditions (Dirichlet)
    facets = mesh.locate_entities_boundary(
        msh, dim=msh.topology.dim - 1, marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], domain_size[0])
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(0.0, dofs, V)

    # Define problem: Poisson equation -Δu = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - domain_size[0] / 2) ** 2 + (x[1] - domain_size[1] / 2) ** 2) / 0.02)
    
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # Solve the problem
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # Save solution
    with io.XDMFFile(msh.comm, "poisson_generalized.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)

    print("Generalized Poisson problem solved and stored.")
    return msh, uh

if __name__ == "__main__":
    poisson_solver()
