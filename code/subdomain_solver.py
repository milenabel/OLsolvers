import os
import numpy as np
import pandas as pd
from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, div

# Path to the general solver CSV
GENERAL_RESULTS_CSV = "../results/general/poisson_error_norms.csv"

class SubdomainSolver:
    def __init__(self, global_mesh_size, global_solution_values, subdomain_bounds, num_elements, results_dir="../results/subdomains"):
        self.global_mesh_size = global_mesh_size  # Store mesh size of the global solver
        self.global_solution_values = global_solution_values  # Store L2, H1, Rel L2 from CSV
        self.subdomain_bounds = subdomain_bounds
        self.num_elements = num_elements
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def solve(self):
        """Solves the Poisson equation on a subdomain using interpolated global solution as boundary conditions."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Create a subdomain mesh using the selected refinement
        msh_sub = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=((self.subdomain_bounds[0], self.subdomain_bounds[2]), 
                    (self.subdomain_bounds[1], self.subdomain_bounds[3])),
            n=self.num_elements,
            cell_type=mesh.CellType.quadrilateral,
        )
        V_sub = fem.functionspace(msh_sub, ("Lagrange", 1))

        u = ufl.TrialFunction(V_sub)
        v = ufl.TestFunction(V_sub)

        # Use the global solution values (read from CSV)
        L2_norm_global, H1_norm_global, relative_L2_global = self.global_solution_values

        # Create a function for Dirichlet boundary conditions
        bc_function = fem.Function(V_sub)
        bc_function.interpolate(lambda x: np.full(x.shape[1], L2_norm_global))  # Fill with L2 norm value

        # Apply Dirichlet BCs using the interpolated function
        facets = mesh.locate_entities_boundary(msh_sub, msh_sub.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V_sub, 1, facets)
        bc = fem.dirichletbc(bc_function, dofs)

        # Weak form with zero source term
        dx_sub = ufl.Measure("dx", domain=msh_sub)
        a = inner(grad(u), grad(v)) * dx_sub
        L = inner(fem.Constant(msh_sub, 0.0), v) * dx_sub

        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        # Compute error norms
        error_L2_form = fem.form(inner(uh - bc_function, uh - bc_function) * dx_sub)
        L2_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(error_L2_form), op=MPI.SUM))

        error_H1_form = fem.form(inner(grad(uh - bc_function), grad(uh - bc_function)) * dx_sub)
        H1_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(error_H1_form), op=MPI.SUM))

        # Compute relative L2 error norm
        projected_L2_form = fem.form(inner(bc_function, bc_function) * dx_sub)
        projected_L2_norm = np.sqrt(msh_sub.comm.allreduce(fem.assemble_scalar(projected_L2_form), op=MPI.SUM))

        relative_L2_error = L2_norm / projected_L2_norm

        print(f"Subdomain {self.subdomain_bounds}, Mesh {self.num_elements} → L2: {L2_norm}, H1: {H1_norm}, Rel L2: {relative_L2_error}")

        # Save results to CSV (only rank 0 writes to file)
        if rank == 0:
            results_filename = os.path.join(self.results_dir, "subdomain_error_norms.csv")
            os.makedirs(os.path.dirname(results_filename), exist_ok=True)

            row = [
                int(self.global_mesh_size[0]), int(self.global_mesh_size[1]),  # Global mesh size
                float(self.subdomain_bounds[0]), float(self.subdomain_bounds[1]),  # Subdomain x bounds
                float(self.subdomain_bounds[2]), float(self.subdomain_bounds[3]),  # Subdomain y bounds
                int(self.num_elements[0]), int(self.num_elements[1]),  # Subdomain mesh size (Nx, Ny)
                float(L2_norm), float(H1_norm), float(relative_L2_error)  # Error norms
            ]

            df = pd.DataFrame([row], columns=[
                "global_mesh_x", "global_mesh_y", 
                "sub_x0", "sub_x1", "sub_y0", "sub_y1", 
                "sub_mesh_x", "sub_mesh_y", 
                "L2_norm", "H1_norm", "Relative_L2_norm"
            ])

            # Debug print before writing
            print(f"Writing to CSV: {results_filename}")
            
            write_header = not os.path.exists(results_filename)
            df.to_csv(results_filename, mode="a", header=write_header, index=False)

            # Debug print after writing
            print(f"Successfully wrote to {results_filename}")

        return msh_sub, uh


if __name__ == "__main__":
    mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (24, 24), (48, 48), (96, 96)]
    subdomain_refinements = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (96, 96), (128, 128), (192, 192)]
    
    # Load the global solver results from CSV
    global_results_df = pd.read_csv(GENERAL_RESULTS_CSV)

    # Convert the dataframe into a dictionary for easy lookup
    global_solutions = {
        (int(row["Nx"]), int(row["Ny"])): (row["L2 Norm"], row["H1 Norm"], row["Relative L2 Norm"])
        for _, row in global_results_df.iterrows()
    }

    # Define the subdomain (currently using only one, but can be expanded)
    subdomains = [(0, 4, 0, 4)]  # (x0, x1, y0, y1)

    for global_mesh_size in mesh_sizes:
        print(f"\n Using global solver results for mesh size {global_mesh_size}")

        # Get the stored global solution values
        global_solution_values = global_solutions[global_mesh_size]

        for subdomain_bounds in subdomains:
            print(f"\n Using subdomain {subdomain_bounds} with boundary conditions from {global_mesh_size}")
            
            for sub_mesh_size in subdomain_refinements:
                print(f"  Refining subdomain with mesh size {sub_mesh_size}")

                sub_solver = SubdomainSolver(global_mesh_size, global_solution_values, subdomain_bounds, sub_mesh_size)
                sub_solver.solve()