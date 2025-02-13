import os
import random
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import compute_closest_entity
from dolfinx.fem import Function
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

        return msh, uh, L2_norm, H1_norm, relative_L2_error, grad(uh)

class SubdomainSolver:
    def __init__(self, global_solver, subdomain_bounds, refinement_factor=2, results_dir="../results/subdomains"):
        self.global_solver = global_solver
        self.subdomain_bounds = subdomain_bounds
        self.refinement_factor = refinement_factor
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Prepare CSV file for saving results
        self.results_file = os.path.join(self.results_dir, "subdomain_error_norms.csv")
    
    def solve(self):
        """Solves the Poisson equation on the subdomain using global solution as boundary conditions."""
        global_msh, global_solution, _, _, _, global_grad = self.global_solver.solve()  # <-- FIXED UNPACKING
        
        subdomain_size = (self.subdomain_bounds[1] - self.subdomain_bounds[0], 
                        self.subdomain_bounds[3] - self.subdomain_bounds[2])
        num_elements = (self.global_solver.num_elements[0] * self.refinement_factor, 
                        self.global_solver.num_elements[1] * self.refinement_factor)
        
        msh = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=((self.subdomain_bounds[0], self.subdomain_bounds[2]),
                    (self.subdomain_bounds[1], self.subdomain_bounds[3])),
            n=num_elements,
            cell_type=mesh.CellType.quadrilateral,
        )
        V = fem.functionspace(msh, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        # Locate boundary nodes and their coordinates
        facets = mesh.locate_entities_boundary(
            msh, dim=msh.topology.dim - 1, marker=lambda x: np.full(x.shape[1], True)
        )
        dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
        boundary_coords = msh.geometry.x[dofs]  # Get boundary node coordinates

        # Ensure the boundary coordinates are in the correct shape (Nx3 for FEniCS)
        boundary_coords_3D = np.column_stack((boundary_coords, np.zeros(boundary_coords.shape[0])))

        # Get closest cells from global mesh
        boundary_cells = np.array([compute_closest_entity(global_msh, c) for c in boundary_coords_3D])

        # Remove invalid points
        valid = boundary_cells >= 0
        boundary_coords_3D = boundary_coords_3D[valid]
        boundary_cells = boundary_cells[valid]

        # Evaluate the global solution at the valid boundary coordinates
        bc_values = np.array([global_solution.eval(c.reshape(1, -1), np.array([cell]))[0] 
                            for c, cell in zip(boundary_coords_3D, boundary_cells)])
        bc = fem.dirichletbc(bc_values, dofs, V)

        # Define weak form
        a = inner(grad(u), grad(v)) * dx
        L = inner(ufl.Constant(0), v) * dx  # No source term for subdomain

        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return msh, uh


if __name__ == "__main__":
    global_solver = GeneralSolver()
    for _ in range(5):  # Run 5 random subdomain tests
        x0, y0 = np.random.uniform(0, 6), np.random.uniform(0, 6)
        x1, y1 = x0 + np.random.uniform(2, 4), y0 + np.random.uniform(2, 4)
        subdomain_bounds = (x0, x1, y0, y1)
        print(f"Running subdomain solver for bounds: {subdomain_bounds}")
        sub_solver = SubdomainSolver(global_solver, subdomain_bounds)
        sub_solver.solve()