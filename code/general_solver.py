import os
import json
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from ufl import dx, grad, inner
import pandas as pd


class GeneralSolver:
    def __init__(self, domain_size=(12.0, 12.0), num_elements=(20, 20), results_dir="../results/general", num_rhs=100):
        self.domain_size = domain_size
        self.num_elements = num_elements
        self.results_dir = results_dir
        self.num_rhs = num_rhs
        os.makedirs(self.results_dir, exist_ok=True)

    def solve(self):
        msh = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [(0.0, 0.0), self.domain_size],
            self.num_elements,
            mesh.CellType.quadrilateral
        )
        V = fem.functionspace(msh, ("Lagrange", 1))

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        # Homogeneous Dirichlet BC
        zero = fem.Function(V)
        boundary_facets = mesh.locate_entities_boundary(
            msh, dim=msh.topology.dim - 1,
            marker=lambda x: np.full(x.shape[1], True)
        )
        bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
        bc = fem.dirichletbc(zero, bc_dofs)

        # Assemble matrix
        a_form = fem.form(inner(grad(u), grad(v)) * dx)
        A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()

        # Create KSP solver once
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()

        f_array = []
        u_array = []
        error_metrics = []

        for i in range(self.num_rhs):
            # Randomized RHS
            coeff_x = np.random.uniform(1, 5)
            coeff_y = np.random.uniform(1, 5)
            f_expr = ufl.sin(coeff_x * ufl.pi * x[0] / self.domain_size[0]) * ufl.sin(coeff_y * ufl.pi * x[1] / self.domain_size[1])
            
            f_func = fem.Function(V)
            f_func.interpolate(lambda x_: np.sin(coeff_x * np.pi * x_[0] / self.domain_size[0]) * 
                                        np.sin(coeff_y * np.pi * x_[1] / self.domain_size[1]))

            # Assemble RHS
            L_form = fem.form(inner(f_func, v) * dx)
            b = fem.petsc.assemble_vector(L_form)
            fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b, [bc])

            # Solve
            u_sol = fem.Function(V)
            x = A.createVecRight()
            solver.solve(b, x)
            u_sol.x.array[:] = x.array



            # Store results
            f_array.append(f_func.x.array.copy())
            u_array.append(u_sol.x.array.copy())

            # For relative error: compare with f_func as "exact" (not really, but good enough)
            u_exact = fem.Function(V)
            u_exact.interpolate(lambda x_: np.sin(coeff_x * np.pi * x_[0] / self.domain_size[0]) * 
                                            np.sin(coeff_y * np.pi * x_[1] / self.domain_size[1]))

            l2_norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(u_sol, u_sol) * dx)), op=MPI.SUM))
            l2_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(u_sol - u_exact, u_sol - u_exact) * dx)), op=MPI.SUM))
            h1_error = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_sol - u_exact), grad(u_sol - u_exact)) * dx)), op=MPI.SUM))
            relative_error = l2_error / l2_norm if l2_norm > 0 else 0.0

            error_metrics.append([i, l2_error, h1_error, relative_error])

        # Save dataset JSON
        dataset = {
            "mesh_size": self.num_elements,
            "domain_size": self.domain_size,
            "u_values": [u.tolist() for u in u_array],  # Convert each np.ndarray to list
            "f_values": [f.tolist() for f in f_array]
        }

        dataset_filename = os.path.join(self.results_dir, f"poisson_dataset_{self.num_elements[0]}x{self.num_elements[1]}.json")
        with open(dataset_filename, "w") as f:
            json.dump(dataset, f)
        print(f"[Saved] Dataset with {self.num_rhs} RHS -> {dataset_filename}")

        # Save CSV error file
        error_df = pd.DataFrame(error_metrics, columns=["Sample", "L2 Error", "H1 Error", "Relative L2 Error"])
        error_csv = os.path.join(self.results_dir, f"poisson_error_norms_{self.num_elements[0]}x{self.num_elements[1]}.csv")
        error_df.to_csv(error_csv, index=False)
        print(f"[Saved] Error metrics -> {error_csv}")


if __name__ == "__main__":
    mesh_sizes = [(3, 3), (4, 4), (6, 6), (8, 8), (12, 12), (20, 20), (24, 24), (48, 48), (96, 96)]
    for size in mesh_sizes:
        print(f"\nRunning solver for mesh: {size}")
        solver = GeneralSolver(num_elements=size, num_rhs=1000)
        solver.solve()

