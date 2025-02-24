import os
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, div
import pandas as pd

class GeneralSolver:
    def __init__(self, domain_size=(12.0, 12.0), num_elements=(12, 12), results_dir="../results/general"):
        self.domain_size = domain_size
        self.num_elements = num_elements
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    
    def solve(self):
        """Solves the Poisson equation and stores results."""
        msh = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=((0.0, 0.0), self.domain_size),
            n=self.num_elements,
            cell_type=mesh.CellType.quadrilateral,
        )
        V = fem.functionspace(msh, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        p_exact_expr = ufl.sin(ufl.pi * x[0] / self.domain_size[0]) * ufl.sin(ufl.pi * x[1] / self.domain_size[1])
        f = -div(grad(p_exact_expr))

        p_exact = fem.Function(V)
        def exact_solution(x):
            return np.sin(np.pi * x[0] / self.domain_size[0]) * np.sin(np.pi * x[1] / self.domain_size[1])
        p_exact.interpolate(exact_solution)

        facets = mesh.locate_entities_boundary(
            msh, dim=msh.topology.dim - 1, marker=lambda x: np.full(x.shape[1], True)
        )
        dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
        bc = fem.dirichletbc(p_exact, dofs)

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        error_L2_form = fem.form(inner(uh - p_exact, uh - p_exact) * dx)
        L2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_form), op=MPI.SUM))

        error_H1_form = fem.form(inner(grad(uh - p_exact), grad(uh - p_exact)) * dx)
        H1_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_H1_form), op=MPI.SUM))

        p_exact_L2_form = fem.form(inner(p_exact, p_exact) * dx)
        p_exact_L2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(p_exact_L2_form), op=MPI.SUM))

        relative_L2_error = L2_norm / p_exact_L2_norm

        solution_filename = os.path.join(self.results_dir, f"poisson_solution_{self.num_elements[0]}x{self.num_elements[1]}.xdmf")
        with io.XDMFFile(MPI.COMM_WORLD, solution_filename, "w") as file:
            file.write_mesh(msh)
            file.write_function(uh)
        
        return msh, uh, L2_norm, H1_norm, relative_L2_error
    
    def load_solution(self):
        """Loads the saved global solution from an XDMF file."""
        solution_filename = os.path.join(self.results_dir, f"poisson_solution_{self.num_elements[0]}x{self.num_elements[1]}.xdmf")
        with io.XDMFFile(MPI.COMM_WORLD, solution_filename, "r") as file:
            msh = file.read_mesh()
            V = fem.functionspace(msh, ("Lagrange", 1))
            uh = fem.Function(V)
            file.read_function(uh)
        return msh, uh

if __name__ == "__main__":
    mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (24, 24), (48, 48), (96, 96)]
    results = []
    
    results_dir = "../results/general"
    os.makedirs(results_dir, exist_ok=True)

    for size in mesh_sizes:
        solver = GeneralSolver(num_elements=size)
        print(f"Running solver for mesh size: {size}")
        _, _, L2_norm, H1_norm, relative_L2_error = solver.solve()
        print(f"Mesh {size}: L2 norm = {L2_norm}, H1 norm = {H1_norm}, Relative L2 norm = {relative_L2_error}\n")

        results.append([size[0], size[1], L2_norm, H1_norm, relative_L2_error])
    
    results_filename = os.path.join(results_dir, "poisson_error_norms.csv")
    df = pd.DataFrame(results, columns=["Nx", "Ny", "L2 Norm", "H1 Norm", "Relative L2 Norm"])
    df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")