import os
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import Function, FunctionSpace
from ufl import dx, grad, inner, div
import pandas as pd

class GeneralSolver:
    def __init__(self, domain_size=(12.0, 12.0), num_elements=(12, 12), results_dir="../results/general"):
        self.domain_size = domain_size
        self.num_elements = num_elements
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def solve(self):
        """Solves the Poisson equation on the full domain and returns the solution."""
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

        # Define exact solution and corresponding source term
        p_exact_expr = ufl.sin(ufl.pi * x[0] / self.domain_size[0]) * ufl.sin(ufl.pi * x[1] / self.domain_size[1])
        f = -div(grad(p_exact_expr))

        p_exact = fem.Function(V)

        def exact_solution(x):
            return np.sin(np.pi * x[0] / self.domain_size[0]) * np.sin(np.pi * x[1] / self.domain_size[1])

        p_exact.interpolate(exact_solution)

        # Apply Dirichlet boundary conditions
        facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V, 1, facets)
        bc = fem.dirichletbc(p_exact, dofs)

        # Define weak form
        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        # Solve system
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return msh, uh


class SubdomainSolver:
    def __init__(self, global_solver, subdomain_bounds, refinement_factor=2, results_dir="../results/subdomains"):
        self.global_solver = global_solver
        self.subdomain_bounds = subdomain_bounds
        self.refinement_factor = refinement_factor
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def interpolate_solution(self, global_solution, V_sub):
        """Interpolates the global solution onto the subdomain function space."""
        projected_function = fem.Function(V_sub)

        def eval_global_solution(x):
            """Evaluates global solution at given subdomain points."""
            values = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                values[i] = global_solution.eval(x[:, i].reshape(1, -1), np.array([0]))[0]
            return values

        projected_function.interpolate(eval_global_solution)
        return projected_function

    import pandas as pd  # Make sure pandas is imported

    def solve(self):
        """Solves the Poisson equation on a subdomain using interpolated global solution as boundary conditions."""
        global_msh, global_solution = self.global_solver.solve()

        # Create a finer subdomain mesh
        num_elements = (self.global_solver.num_elements[0] * self.refinement_factor, 
                        self.global_solver.num_elements[1] * self.refinement_factor)

        msh_sub = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=((self.subdomain_bounds[0], self.subdomain_bounds[2]), 
                    (self.subdomain_bounds[1], self.subdomain_bounds[3])),
            n=num_elements,
            cell_type=mesh.CellType.quadrilateral,
        )
        V_sub = fem.functionspace(msh_sub, ("Lagrange", 1))

        u = ufl.TrialFunction(V_sub)
        v = ufl.TestFunction(V_sub)

        # Interpolate global solution onto the subdomain
        projected_solution = self.interpolate_solution(global_solution, V_sub)

        # Apply Dirichlet BCs using projected solution
        facets = mesh.locate_entities_boundary(msh_sub, msh_sub.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V_sub, 1, facets)
        bc = fem.dirichletbc(projected_solution, dofs)

        # Weak form with zero source term
        dx_sub = ufl.Measure("dx", domain=msh_sub)
        a = inner(grad(u), grad(v)) * dx_sub
        L = inner(fem.Constant(msh_sub, 0.0), v) * dx_sub

        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # Compute error norms
        error_L2_form = fem.form(inner(uh - projected_solution, uh - projected_solution) * dx_sub)
        L2_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(error_L2_form), op=MPI.SUM))

        error_H1_form = fem.form(inner(grad(uh - projected_solution), grad(uh - projected_solution)) * dx_sub)
        H1_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(error_H1_form), op=MPI.SUM))

        # Compute relative L2 error norm
        projected_L2_form = fem.form(inner(projected_solution, projected_solution) * dx_sub)
        projected_L2_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(projected_L2_form), op=MPI.SUM))

        relative_L2_error = L2_norm / projected_L2_norm

        # Save results with a clean format
        results_filename = os.path.join(self.results_dir, "subdomain_error_norms.csv")

        df = pd.DataFrame(
            [[self.global_solver.num_elements[0], self.global_solver.num_elements[1], 
            self.subdomain_bounds[0], self.subdomain_bounds[1], 
            self.subdomain_bounds[2], self.subdomain_bounds[3], 
            L2_norm, H1_norm, relative_L2_error]],
            columns=["mesh_x", "mesh_y", "sub_x0", "sub_x1", "sub_y0", "sub_y1", "L2_norm", "H1_norm", "Relative_L2_norm"]
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)

        # Check if file exists; write header only if it doesn’t exist
        write_header = not os.path.exists(results_filename)

        # Write CSV and force flush
        with open(results_filename, "a", newline="") as f:
            df.to_csv(f, mode="a", header=write_header, index=False)
            f.flush()  # Ensure data is written immediately
            os.fsync(f.fileno())  # Ensure file changes are committed to disk
        
        print(f"Data saved: {results_filename}")

        print(f"Global mesh {self.global_solver.num_elements}: Subdomain {self.subdomain_bounds} → L2: {L2_norm}, H1: {H1_norm}, Rel L2: {relative_L2_error}")

        # Save solution
        solution_filename = os.path.join(self.results_dir, f"subdomain_solution_{self.global_solver.num_elements}_{num_elements[0]}x{num_elements[1]}.xdmf")
        with io.XDMFFile(MPI.COMM_WORLD, solution_filename, "w") as file:
            file.write_mesh(msh_sub)
            file.write_function(uh)

        return msh_sub, uh




if __name__ == "__main__":
    mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (24, 24), (48, 48), (96, 96)]

    for mesh_size in mesh_sizes:
        print(f"\n Running global solver for mesh size {mesh_size}")
        global_solver = GeneralSolver(num_elements=mesh_size)

        for _ in range(5):  # Run 5 random subdomain tests per mesh size
            x0, y0 = np.random.uniform(0, 6), np.random.uniform(0, 6)
            x1, y1 = x0 + np.random.uniform(2, 4), y0 + np.random.uniform(2, 4)
            subdomain_bounds = (x0, x1, y0, y1)

            print(f"Subdomain {subdomain_bounds}")
            sub_solver = SubdomainSolver(global_solver, subdomain_bounds)
            sub_solver.solve()
